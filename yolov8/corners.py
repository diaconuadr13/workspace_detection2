import time

import cv2
import cv2 as cv
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pyrealsense2 as rs
import os


def load_model():
    model = YOLO('C:\\Users\\Adi\\Desktop\\licenta_buna\\yolov8\\train11\\weights\\best.pt')
    return model


def get_results(model, path):
    results = model(path)
    # Visualize the results
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Show results to screen (in supported environments)
        r.show()

        # Save results to disk
        r.save(filename=f'results{i}.jpg')
    return results


def get_corners_poly(results):
    corners = []
    angle = 0
    center = 0
    for r in results:
        if r.masks:
            points = r.masks.xy
            contours = np.array(points).reshape((-1, 1, 2)).astype(np.int32)

            # Approximate the contour to a polygon
            epsilon = 0.01 * cv2.arcLength(contours, True)
            approx = cv2.approxPolyDP(contours, epsilon, True)

            # Calculate the minimum area rectangle that encloses the contour
            rect = cv2.minAreaRect(contours)

            # Get the four vertices of the rectangle
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # round to integer values

            center = (int(rect[0][0]), int(rect[0][1]))
            width = int(rect[1][0])
            height = int(rect[1][1])
            angle = float(rect[2])

            if width < height:
                angle = 90 - angle
            else:
                angle = 180 - angle

            # Add the corner points of the polygon to the corners list
            for point in approx:
                corners.append(tuple(point[0]))  # unpack the point from the list

    return corners, angle, center


def get_corners_minarea_poly(results):
    corners = []
    angle = 0
    center = 0
    for r in results:
        if results[0].masks:
            points = r.masks.xy
            contours = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
            # Calculate the minimum area rectangle that encloses the contour
            rect = cv2.minAreaRect(contours)

            # Get the four vertices of the rectangle
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # round to integer values

            center = (int(rect[0][0]), int(rect[0][1]))
            width = int(rect[1][0])
            height = int(rect[1][1])
            angle = float(rect[2])

            if width < height:
                angle = 90 - angle
            else:
                angle = 180 - angle

            # Approximate the contour to a polygon
            epsilon = 0.01 * cv2.arcLength(contours, True)
            approx = cv2.approxPolyDP(contours, epsilon, True)

            # For each vertex of the rectangle, find the point in the approximated polygon that is closest
            for vertex in box:
                min_distance = float('inf')
                closest_point = None
                for point in approx:
                    distance = np.linalg.norm(point[0] - vertex)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = point[0]

                # Add the closest point to the corners list
                corners.append(closest_point)
    print(corners)
    return corners, angle, center


def get_corners_minarea(results):
    corners = []
    angle = 0
    center = 0
    for r in results:
        if r.masks:
            points = r.masks.xy
            contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.intp(box)

            center = (int(rect[0][0]), int(rect[0][1]))
            width = int(rect[1][0])
            height = int(rect[1][1])
            angle = float(rect[2])

            if width < height:
                angle = 90 - angle
            else:
                angle = 180 - angle
            for (x, y) in box:
                corners.append((x, y))

    return corners, angle, center


def draw_on_image(img, corners, angle, center):
    cv.circle(img, corners, 3, (0, 0, 255), -1)
    cv.circle(img, center, 3, (0, 255, 0), -1)
    label = "  Rotation Angle: " + str(angle) + " degrees"
    cv.imshow('img', img)
    cv.imwrite('corners.png', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_top_left_corner(corners):
    max_x = max(corner[0] for corner in corners)
    max_y = max(corner[1] for corner in corners)
    print(f"Top left corner: ({max_x}, {max_y})")
    return max_x, max_y


img = cv2.imread('C:\\Users\\Adi\\Desktop\\test\\poza_Color.png')
result = get_results(load_model(), img)
corners, angle, center = get_corners_minarea_poly(result)
corner = (get_top_left_corner(corners))
draw_on_image(img, corners[2], angle, center)