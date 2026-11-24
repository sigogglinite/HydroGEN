"""
make_icon.py - draws the HydroGEN icon: a muscular water molecule (blue O) flexing its
bond-arms, with orange H fists - wearing sunglasses and a smirk. Writes HydroGEN.ico (16-256 px) and HydroGEN.png.

    python make_icon.py          (needs Pillow; Anaconda has it)
"""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1024                                   # drawn large, then scaled down (smooth edges)

O_FILL, O_EDGE, O_SHINE = (37, 99, 235, 255), (23, 37, 84, 255), (147, 197, 253, 255)
ARM_FILL, ARM_EDGE, ARM_SHADE = (226, 232, 240, 255), (30, 41, 59, 255), (186, 196, 210, 255)
FIST_FILL, FIST_EDGE, FIST_SHADE = (249, 115, 22, 255), (124, 45, 18, 255), (234, 88, 12, 255)
INK = (15, 23, 42, 255)
WHITE = (255, 255, 255, 255)


def mx(x, side):
    return x if side > 0 else S - x


def arm(d, side, edge):
    """Upper arm out from the body, forearm up, bicep bump on top of the upper arm."""
    shoulder, elbow, wrist = (330, 650), (120, 610), (150, 330)
    pts = [(mx(x, side), y) for x, y in (shoulder, elbow, wrist)]
    col = ARM_EDGE if edge else ARM_FILL
    w = 120 if edge else 88
    d.line(pts, fill=col, width=w, joint="curve")
    for x, y in pts:
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=col)
    # bicep: big bump rising from the upper arm
    bx, by, bw, bh = mx(235, side), 545, 250, 210
    p = 16 if edge else 0
    d.ellipse([bx - bw / 2 - p, by - bh / 2 - p, bx + bw / 2 + p, by + bh / 2 + p], fill=col)
    if not edge:
        # shading under the bump and a shine on top
        d.chord([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2], 20, 160, fill=ARM_SHADE)
        d.ellipse([bx - bw * 0.30, by - bh * 0.36, bx - bw * 0.02, by - bh * 0.14], fill=WHITE)


def fist(d, side):
    """Front view of a clenched fist: four knuckles on top, finger lines, thumb across."""
    cx, cy = mx(165, side), 235
    w, h = 230, 190
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    e = 20
    d.rounded_rectangle([x0 - e, y0 - e, x1 + e, y1 + e], radius=70, fill=FIST_EDGE)
    kr = w / 8
    for k in range(4):                                  # knuckle bumps (outline pass)
        kx = x0 + kr + k * 2 * kr
        d.ellipse([kx - kr - e, y0 - kr - e, kx + kr + e, y0 + kr + e], fill=FIST_EDGE)
    d.rounded_rectangle([x0, y0, x1, y1], radius=60, fill=FIST_FILL)
    for k in range(4):
        kx = x0 + kr + k * 2 * kr
        d.ellipse([kx - kr, y0 - kr, kx + kr, y0 + kr], fill=FIST_FILL)
    for k in range(1, 4):                               # finger separations
        lx = x0 + k * 2 * kr
        d.line([(lx, y0 - kr * 0.4), (lx, cy + 5)], fill=FIST_EDGE, width=14)
    # thumb lying across the fingers
    tx0, tx1 = (x0 + 20, cx + 60) if side > 0 else (cx - 60, x1 - 20)
    d.rounded_rectangle([tx0, cy + 5, tx1, cy + 70], radius=32, fill=FIST_SHADE, outline=FIST_EDGE, width=14)


def draw():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for edge in (True, False):
        arm(d, 1, edge)
        arm(d, -1, edge)
    for side in (1, -1):                                # flex marks
        cx = mx(235, side)
        for dx, top in ((-95, 395), (0, 370), (95, 395)):
            x = cx + dx
            d.line([(x, top), (x + 0.25 * dx, top - 55)], fill=INK, width=16)
    fist(d, 1)
    fist(d, -1)

    cx, cy, r = 512, 700, 225                           # O body
    d.ellipse([cx - r - 26, cy - r - 26, cx + r + 26, cy + r + 26], fill=O_EDGE)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=O_FILL)
    d.ellipse([cx - 150, cy - 190, cx - 40, cy - 115], fill=O_SHINE)
    sunglasses(d, cx, cy)
    smirk(d, cx, cy)
    return img


LENS, FRAME, GLINT = (17, 24, 39, 255), (2, 6, 23, 255), (148, 163, 184, 255)


def sunglasses(d, cx, cy):
    """Wraparound shades: two wide lenses (flat top, rounded bottom), a bridge, temple arms to the
    edge of the face, and a diagonal glint on each lens."""
    d.line([(cx - 232, cy - 70), (cx - 175, cy - 62)], fill=FRAME, width=26)     # temple arms
    d.line([(cx + 175, cy - 62), (cx + 232, cy - 70)], fill=FRAME, width=26)
    d.line([(cx - 40, cy - 58), (cx + 40, cy - 58)], fill=FRAME, width=26)       # bridge
    for sgn in (-1, 1):
        x_in, x_out = cx + sgn * 22, cx + sgn * 185
        xl, xr = min(x_in, x_out), max(x_in, x_out)
        top, bot = cy - 92, cy + 22
        lens = [(xl, top), (xr, top), (xr - sgn * 12 if sgn < 0 else xr, top + 60),
                (xl + 8 if sgn < 0 else xl + 12, top + 60)]
        d.rounded_rectangle([xl - 14, top - 14, xr + 14, bot + 14], radius=56, fill=FRAME)   # frame
        d.rounded_rectangle([xl - 14, top - 14, xr + 14, top + 40], radius=14, fill=FRAME)   # flat top
        d.rounded_rectangle([xl, top, xr, bot], radius=46, fill=LENS)
        d.rectangle([xl, top, xr, top + 30], fill=LENS)
        gx = (xl + xr) / 2 - 20                                                       # glint
        d.line([(gx - 30, top + 74), (gx + 22, top + 14)], fill=GLINT, width=16)
        d.line([(gx + 14, top + 88), (gx + 50, top + 46)], fill=GLINT, width=9)
    _ = lens


def smirk(d, cx, cy):
    """Lopsided grin: flat on the left, curling up on the right, with a little dimple."""
    p0, p1, p2 = (cx - 95, cy + 108), (cx + 40, cy + 140), (cx + 112, cy + 60)    # quadratic Bezier
    pts = [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
            (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in [k / 60 for k in range(61)]]
    w = 26
    for x, y in pts:                                     # round brush along the curve: smooth stroke
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=INK)
    d.line([(cx + 100, cy + 96), (cx + 134, cy + 104)], fill=INK, width=14)         # dimple



if __name__ == "__main__":
    big = draw()
    big.resize((512, 512), Image.LANCZOS).save(os.path.join(HERE, "HydroGEN.png"))
    big.resize((256, 256), Image.LANCZOS).save(
        os.path.join(HERE, "HydroGEN.ico"),
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)])
    print("wrote HydroGEN.ico and HydroGEN.png")
