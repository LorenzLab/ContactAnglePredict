from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
from scipy import spatial

img_path = "test_pics/6.jpg"


def find_intersect(circle, line, return_mode="xy"):
    """
    Find the intersection points of a circle and a line.
    circle: (x, y, r)
    line: (rho, theta)
    return_mode: 'xy' or 'angle', if 'angle', the start angle and end angle is theta +- offset
    """

    circle_x, circle_y, circle_r = circle
    rho, theta = line

    # calculate the intersection point
    cosv = (rho - circle_x * math.cos(theta) - circle_y * math.sin(theta)) / circle_r
    if not (-1 <= cosv <= 1):
        # no intersection
        return None
    offset = math.acos(cosv)  # / math.pi * 180
    if return_mode == "angle":
        return offset

    x1 = circle_x + circle_r * math.cos(theta + offset)
    y1 = circle_y + circle_r * math.sin(theta + offset)
    x2 = circle_x + circle_r * math.cos(theta - offset)
    y2 = circle_y + circle_r * math.sin(theta - offset)
    return x1, y1, x2, y2


def find_upper_or_lower_curve(image, circle, line, fitting_points=20):
    """
    image: the edge image
    circle: (x, y, r)
    line: (rho, theta)
    """
    circle_x, circle_y, circle_r = circle
    rho, theta = line
    edge_points = np.argwhere(image > 0)[:, ::-1]

    # find intersection angle
    offset = find_intersect(circle, line, return_mode="angle")
    angle1 = (theta + offset + 2 * math.pi) % (2 * math.pi)
    angle2 = (theta - offset + 2 * math.pi) % (2 * math.pi)

    angle_big, angle_small = max(angle1, angle2), min(angle1, angle2)
    totlen1 = 0
    totlen2 = 0

    kd_tree_finder = spatial.KDTree(edge_points.astype(np.float32))

    if offset is None:
        return None
    for i in range(1, fitting_points):
        angle = angle_big + (angle_small - angle_big) / fitting_points * i
        x = circle_x + circle_r * math.cos(angle)
        y = circle_y + circle_r * math.sin(angle)
        distance, index = kd_tree_finder.query([x, y], k=1)
        nearest_point = edge_points[index]
        totlen1 += distance
        # cv2.ellipse(image, (int(circle_x), int(circle_y)), (int(circle_r), int(circle_r)), 0, angle1 / math.pi * 180, angle2 / math.pi * 180, 255, 2)

    for i in range(1, fitting_points):
        angle = (
            angle_small + (-2 * math.pi - angle_small + angle_big) / fitting_points * i
        )
        x = circle_x + circle_r * math.cos(angle)
        y = circle_y + circle_r * math.sin(angle)
        distance, index = kd_tree_finder.query([x, y], k=1)
        nearest_point = edge_points[index]
        totlen2 += distance
        # cv2.ellipse(image, (int(circle_x), int(circle_y)), (int(circle_r), int(circle_r)), 0, angle2 / math.pi * 180, angle1 / math.pi * 180 - 360, 255, 2)

    if totlen1 < totlen2:
        return angle_big / math.pi * 180, angle_small / math.pi * 180, angle1, angle2
    else:
        return (
            angle_small / math.pi * 180,
            angle_big / math.pi * 180 - 360,
            angle1,
            angle2,
        )


kernel_size = 5
low_threshold = 95
high_threshold = 100
blur_times = 2

image = cv2.imread(img_path, cv2.IMREAD_COLOR)
image = cv2.resize(image, (880, 880))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
linear_filter = image_gray
for i in range(blur_times):
    linear_filter = cv2.GaussianBlur(linear_filter, (kernel_size, kernel_size), 0)
    # image = cv2.convertScaleAbs(image, alpha=1.3, beta=0)

ret3, thresh = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(thresh, low_threshold, high_threshold)

cv2.imshow("edges", edges)
cv2.waitKey(0)

# Detecting lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)


# Detecting circles
detected_circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,
    1,
    200000,
    param1=20,
    param2=8,
    minRadius=60,
    maxRadius=10000,
)

if lines is None:
    raise ValueError("No line detected")

if detected_circles is None:
    raise ValueError("No circle detected")

# Draw the lines
rho, theta = lines[0][0]
a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho
x1 = int(x0 + 1000 * (-b))
y1 = int(y0 + 1000 * (a))
x2 = int(x0 - 1000 * (-b))
y2 = int(y0 - 1000 * (a))
cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

detected_circles = np.uint16(np.around(detected_circles))

for pt in detected_circles[0, :]:
    a, b, r = pt[0], pt[1], pt[2]


tangent_line_length = 50

# Find the intersection points
circle = detected_circles[0][0]
line = lines[0][0]
x1, y1, x2, y2 = find_intersect(circle, line)
angle_s, angle_e, angle1, angle2 = find_upper_or_lower_curve(
    edges, detected_circles[0][0], lines[0][0]
)
cv2.ellipse(
    image,
    (int(circle[0]), int(circle[1])),
    (int(circle[2]), int(circle[2])),
    0,
    angle_s,
    angle_e,
    (0, 255, 0),
    2,
)
cv2.circle(image, (int(x1), int(y1)), 5, (0, 0, 255), 3)
cv2.circle(image, (int(x2), int(y2)), 5, (0, 0, 255), 3)
cv2.line(
    image,
    (
        int(x1 - math.sin(angle1) * tangent_line_length),
        int(y1 + math.cos(angle1) * tangent_line_length),
    ),
    (
        int(x1 + math.sin(angle1) * tangent_line_length),
        int(y1 - math.cos(angle1) * tangent_line_length),
    ),
    (0, 0, 255),
    2,
)
cv2.line(
    image,
    (
        int(x2 - math.sin(angle2) * tangent_line_length),
        int(y2 + math.cos(angle2) * tangent_line_length),
    ),
    (
        int(x2 + math.sin(angle2) * tangent_line_length),
        int(y2 - math.cos(angle2) * tangent_line_length),
    ),
    (0, 0, 255),
    2,
)

angle_drop_line = abs(
    math.atan(
        (math.tan(line[1]) - math.tan(angle1))
        / (math.tan(line[1]) * math.tan(angle1) + 1)
    )
)
if abs(angle1 - angle2) > math.pi:
    angle_drop_line = math.pi - angle_drop_line
print(
    "The angle between the line and the tangent line is: {} deg".format(
        angle_drop_line / math.pi * 180
    )
)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.title("angle: {} deg".format(angle_drop_line / math.pi * 180))
# cv2.imshow('edges', edges)
plt.show()
