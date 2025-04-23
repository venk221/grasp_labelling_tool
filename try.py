import cv2
import numpy as np
import os

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f">>> Clicked point: {clicked_point}")

def find_nearest_edge_point(contours, point):
    """ Find the nearest contour point to the clicked point. """
    min_dist = float('inf')
    closest = None
    for contour in contours:
        for pt in contour:
            dist = np.linalg.norm(np.array(pt[0]) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                closest = tuple(pt[0])
    return closest

def compute_tangent_at_point(contour, target_point):
    """ Compute the tangent angle at the given point on the contour. """
    contour = np.squeeze(contour)
    idx = np.argmin(np.linalg.norm(contour - target_point, axis=1))
    prev_idx = (idx - 1) % len(contour)
    next_idx = (idx + 1) % len(contour)
    delta = contour[next_idx] - contour[prev_idx]
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    return angle

def draw_grasp_box(image, center, angle, width=10, height=40, color=(0, 255, 0)):
    """ Draw a grasp rectangle centered at the point with given angle and size. """
    rect = ((center[0], center[1]), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, color, 2)
    return box

def main():
    rgb_path = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/mug/5_rgb.png"
    depth_path = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/mug/5_depth.tiff"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if rgb is None or depth is None:
        print("Error loading images.")
        return

    print(">>> Click on object...")
    cv2.namedWindow("Click on Object")
    cv2.setMouseCallback("Click on Object", mouse_callback)
    cv2.imshow("Click on Object", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if clicked_point is None:
        print("No click received.")
        return

    x, y = clicked_point
    center_depth = float(depth[y, x])
    print(f">>> Clicked: ({x}, {y}), depth: {center_depth:.6f}")

    # Create depth mask
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[(depth > center_depth - 0.01) & (depth < center_depth + 0.01)] = 255

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return

    nearest_point = find_nearest_edge_point(contours, clicked_point)
    print(f">>> Nearest edge point: {nearest_point}")

    # Find which contour the nearest point belongs to
    selected_contour = None
    for cnt in contours:
        if nearest_point in cnt.squeeze().tolist():
            selected_contour = cnt
            break
    if selected_contour is None:
        selected_contour = contours[0]  # fallback

    angle = compute_tangent_at_point(selected_contour, nearest_point)
    print(f">>> Tangent angle: {angle:.2f}°")

    # Draw grasp
    grasp_box = draw_grasp_box(rgb, clicked_point, angle)

    # Save result
    result_path = os.path.join(output_dir, "grasp_result.png")
    cv2.imwrite(result_path, rgb)
    print(f">>> Grasp saved to: {result_path}")

    # Show it
    cv2.imshow("Grasp Result", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

