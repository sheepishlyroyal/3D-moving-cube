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
#imp0
def rotate_point(x, y, z, angle_x, angle_y, angle_z):
    angle_x, angle_y, angle_z = map(math.radians, [angle_x, angle_y, angle_z])

    x_rot = x
    y_rot = y * math.cos(angle_x) - z * math.sin(angle_x)
    z_rot = y * math.sin(angle_x) + z * math.cos(angle_x)

    x_rot2 = x_rot * math.cos(angle_y) + z_rot * math.sin(angle_y)
    y_rot2 = y_rot
    z_rot2 = -x_rot * math.sin(angle_y) + z_rot * math.cos(angle_y)

    x_final = x_rot2 * math.cos(angle_z) - y_rot2 * math.sin(angle_z)
    y_final = x_rot2 * math.sin(angle_z) + y_rot2 * math.cos(angle_z)
    z_final = z_rot2 

    return round(x_final), round(y_final), round(z_final)

def project_to_2d(x, y, z, cx, cy, depth=20):
    scale = depth / (depth + z) if (depth + z) > 0 else 0.1
    return round(x * scale + cx), round(y * scale + cy)

def points_draw(x1, y1, x2, y2, steps=20):
    points = np.linspace((x1, y1), (x2, y2), steps).astype(int)
    return [(x, y) for x, y in points]

def draw_cube(vertices, edges, terminal_width, terminal_height):
    grid = [[' ' for _ in range(terminal_width)] for _ in range(terminal_height)]
    for v1, v2 in edges:
        x1, y1 = vertices[v1]
        x2, y2 = vertices[v2]
        points = points_draw(x1, y1, x2, y2, steps=20)

        for i, (x, y) in enumerate(points):
            if 0 <= x < terminal_width and 0 <= y < terminal_height:
                grid[y][x] = '.' if i == 0 or i == len(points) - 1 else '|' if x == points[0][0] or x == points[-1][0] else '_' if y == points[0][-1] or y == points[0][0] else '.'

    print('\n'.join(''.join(row) for row in grid))
#imp1
def main():
    size = 8
    terminal_width, terminal_height = get_terminal_size()
    cx, cy = terminal_width // 2, terminal_height // 2

    vertices = [
        (-size, -size, -size), (size, -size, -size), (size, size, -size), (-size, size, -size),
        (-size, -size, size), (size, -size, size), (size, size, size), (-size, size, size)
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


    detector = dlib.get_frontal_face_detector()
    predictor = None
    try:
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    except Exception as e:
        print(f"{e}")
        exit()  

    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("snakeoil 404 lol")
        exit()

    #imp3
    angle_x = angle_y = angle_z = 0
    SCALE_X = 0.5
    SCALE_Y = 0.3
    MIN_MOVEMENT = 1

    prev_face_center_x = prev_face_center_y = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("come in view bro")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if faces:
            face = faces[0] 
            landmarks = predictor(gray, face)
            nose = landmarks.part(30)  

            face_center_x, face_center_y = nose.x, nose.y

            if prev_face_center_x is None or prev_face_center_y is None:
                prev_face_center_x, prev_face_center_y = face_center_x, face_center_y

            delta_x = abs(face_center_x - prev_face_center_x)
            delta_y = abs(face_center_y - prev_face_center_y)

            if delta_x > MIN_MOVEMENT:
                angle_y = (frame.shape[1] // 2 - face_center_x) * SCALE_Y

            if delta_y > MIN_MOVEMENT:
                angle_x = (frame.shape[0] // 2 - face_center_y) * SCALE_X

            prev_face_center_x, prev_face_center_y = face_center_x, face_center_y

        clear_screen()
        rotated_vertices = [rotate_point(vx, vy, vz, angle_x, angle_y, angle_z) for vx, vy, vz in vertices]
        projected_vertices = [project_to_2d(vx, vy, vz, cx, cy) for vx, vy, vz in rotated_vertices]
        draw_cube(projected_vertices, edges, terminal_width, terminal_height)

        cv2.imshow("cam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
