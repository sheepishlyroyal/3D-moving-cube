import os
import math
import shutil
import cv2
import dlib
import numpy as np

def get_terminal_size():
    return shutil.get_terminal_size()

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def rotate_point(x, y, z, ax, ay, az):
    ax, ay, az = map(math.radians, [ax, ay, az])

    xr = x
    yr = y * math.cos(ax) - z * math.sin(ax)
    zr = y * math.sin(ax) + z * math.cos(ax)

    xr2 = xr * math.cos(ay) + zr * math.sin(ay)
    yr2 = yr
    zr2 = -xr * math.sin(ay) + zr * math.cos(ay)

    xf = xr2 * math.cos(az) - yr2 * math.sin(az)
    yf = xr2 * math.sin(az) + yr2 * math.cos(az)
    zf = zr2 

    return round(xf), round(yf), round(zf)

def project_to_2d(x, y, z, cx, cy, depth=20):
    scale = depth / (depth + z) if (depth + z) > 0 else 0.1
    return round(x * scale + cx), round(y * scale + cy)

def points_draw(x1, y1, x2, y2, steps=20):
    points = np.linspace((x1, y1), (x2, y2), steps).astype(int)
    return [(x, y) for x, y in points]

def draw_cube(verts, edges, tw, th):
    grid = [[' ' for _ in range(tw)] for _ in range(th)]
    for v1, v2 in edges:
        x1, y1 = verts[v1]
        x2, y2 = verts[v2]
        points = points_draw(x1, y1, x2, y2, steps=20)

        for i, (x, y) in enumerate(points):
            if 0 <= x < tw and 0 <= y < th:
                grid[y][x] = '.' if i == 0 or i == len(points) - 1 else '|' if x == points[0][0] or x == points[-1][0] else '_' if y == points[0][-1] or y == points[0][0] else '.'

    print('\n'.join(''.join(row) for row in grid))

def main():
    size = 8
    tw, th = get_terminal_size()
    cx, cy = tw // 2, th // 2

    verts = [
        (-size, -size, -size), (size, -size, -size), (size, size, -size), (-size, size, -size),
        (-size, -size, size), (size, -size, size), (size, size, size), (-size, size, size)
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

    det = dlib.get_frontal_face_detector()
    pred = None
    try:
        pred = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    except Exception as e:
        print(f"{e}")
        exit()  

    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("snakeoil 404 lol")
        exit()

    ax = ay = az = 0
    SCALE_X = 0.5
    SCALE_Y = 0.3
    MIN_MOVEMENT = 1

    pfcx = pfcy = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("come in view bro")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = det(gray)

        if faces:
            face = faces[0] 
            landmarks = pred(gray, face)
            nose = landmarks.part(30)  

            fcx, fcy = nose.x, nose.y

            if pfcx is None or pfcy is None:
                pfcx, pfcy = fcx, fcy

            dx = abs(fcx - pfcx)
            dy = abs(fcy - pfcy)

            if dx > MIN_MOVEMENT:
                ay = (frame.shape[1] // 2 - fcx) * SCALE_Y

            if dy > MIN_MOVEMENT:
                ax = (frame.shape[0] // 2 - fcy) * SCALE_X

            pfcx, pfcy = fcx, fcy

        clear_screen()
        rverts = [rotate_point(vx, vy, vz, ax, ay, az) for vx, vy, vz in verts]
        pverts = [project_to_2d(vx, vy, vz, cx, cy) for vx, vy, vz in rverts]
        draw_cube(pverts, edges, tw, th)

        cv2.imshow("cam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
