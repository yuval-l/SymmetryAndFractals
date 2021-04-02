import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import math
from scipy import ndimage

# parser = argparse.ArgumentParser(description="*632 symmetry signature detector")
# parser.add_argument('-i', '--input', help='input image')
# parser.add_argument('-g', '--gaussian', default=3, help='gaussian filter factor')
# parser.add_argument('-k', '--kernel', default=3, help='morphology transformation kernel')
# parser.add_argument('-c', '--adaptiveThConst', default=-1, help='Adaptive threshold constant')
# parser.add_argument('-b', '--buffer', default=15, help='tile buffer')
# parser.add_argument('-m3', '--move3ratio', default=0, help='move *3 (ratio to buffer)')
# parser.add_argument('-m2', '--move2ratio', default=0, help='move *2 (ratio to buffer)')
# parser.add_argument('-s', '--similarity_th', default=0.8, help='similarity threshold')
# args = parser.parse_args()
#
# img_name = args.input
# gaussian_factor = args.gaussian
# morphology_kernel = args.kernel
# adapt_th_const = args.adaptiveThConst
# buffer = args.buffer
# move_3_rat = args.move3ratio
# move_2_rat = args.move2ratio
# similarity_th = args.similarity_th
#
# img_name = 'C:/Users/youva/OneDrive/Desktop/CS-HIT/3rd-year/semester-A/symmetry-and-fractals/test/star632.jpg'
# gaussian_factor = 9
# morphology_kernel = 9
# adapt_th_const = -1
# buffer = 15
# move_3_rat = 0.6
# similarity_th = 0.8
# move_2_rat = 0
# move_3_rat = 0


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return len(lst3)


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def crop_minAreaRect(img, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    if (angle == -90):
        angle = 0
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop


def sim_in_th(sim_val_score, axis, text):
    print(f'{text} axis similarity: {sim_val_score}')

    if (sim_val_score > 0) and (sim_val_score > similarity_th):
        axis = True
        print(f'Setting {text} axis reflection to True')
        print()
        return True
    elif (sim_val_score < 0 and (abs(sim_val_score)) < 1 - similarity_th):
        axis = True
        print(f'Setting {text} axis reflection to True')
        print()
        return True
    else:
        print(f'{text} axis reflection wasn\'t found')
        print()
        return False


# check 0,30,60,90,120,150 degrees axis
def axis_reflective_test(crop_img, theta, rot_rect):
    h = crop_img.shape[0]
    w = crop_img.shape[1]

    (x, y), (w, h), ang = rot_rect
    Cx, Cy = int(w / 2), int(h / 2)

    cent = 20, 20
    mask = np.zeros_like(crop_img)
    theta1 = (theta * np.pi / 180)
    length = max(h, w) * 2
    x_1 = int(Cx - length * np.cos(theta1))
    y_1 = int(Cy - length * np.sin(theta1))
    x_2 = int(Cx + length * np.cos(theta1))
    y_2 = int(Cy + length * np.sin(theta1))
    mask = cv2.line(mask, (x_1, y_1), (x_2, y_2), 255, 2)
    cv2.floodFill(mask, None, (cent[0], cent[1]), (255, 255, 255))

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f"Axis {theta} degrees reflection test", fontsize=20)
    crop_img_d1 = crop_img.copy()
    crop_img_d2 = crop_img.copy()
    crop_img_d1[mask == 255] = 255
    crop_img_d2[mask == 0] = 255

    # plt.subplot(1, 3, 1)
    # plt.imshow(crop_img_d1)
    # plt.title("first half", fontsize=16)
    # plt.subplot(1, 3, 2)
    # plt.imshow(crop_img_d2)
    # plt.title("second half", fontsize=16)

    crop_img_d2 = cv2.flip(crop_img_d2, -1)
    plt.subplot(1, 3, 3)
    plt.imshow(crop_img_d2)
    plt.title("second half mirrored", fontsize=16)

    d_sim = cv2.matchTemplate(crop_img_d1, crop_img_d2, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(d_sim)
    return max_val


def crop_img_center(img, ref):
    width, height = img.shape[1], img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(ref.shape[0] / 3), int(ref.shape[1] / 3)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img


def crop_ptrn(img, ref):
    width, height = img.shape[1], img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    crop_img = img[mid_x - int(mid_x / 1.5):mid_x + int(mid_x / 1.5), mid_y - int(mid_y / 1.5):mid_y + int(mid_y / 1.5)]
    return crop_img


def check_3_rotations(lattice_id, lattice_img):
    rotated120_1_0 = ndimage.rotate(lattice_img, 60)
    rotated120_1 = ndimage.rotate(rotated120_1_0, 60)
    rotated240_1_0 = ndimage.rotate(lattice_img, 120)
    rotated240_1 = ndimage.rotate(rotated240_1_0, 120)
    rotated0_1 = ndimage.rotate(lattice_img, 0)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"3-rotation similairty test - lattice #{lattice_id}", fontsize=20)
    cropped_img_0 = crop_img_center(rotated0_1, rotated0_1)
    cropped_img_120 = crop_img_center(rotated120_1, rotated0_1)
    cropped_img_240 = crop_img_center(rotated240_1, rotated0_1)

    # plt.subplot(1, 3, 1)
    # plt.imshow(cropped_img_0)
    # plt.title("original")
    # plt.subplot(1, 3, 2)
    # plt.imshow(cropped_img_120)
    # plt.title("120 degrees rotated")
    # plt.subplot(1, 3, 3)
    # plt.imshow(cropped_img_240)
    # plt.title("240 degrees rotated")
    # plt.show()

    cropped_img_0 = crop_ptrn(cropped_img_0, cropped_img_0)
    sim_score_1 = cv2.matchTemplate(cropped_img_120, cropped_img_0, cv2.TM_CCOEFF_NORMED)
    sim_score_2 = cv2.matchTemplate(cropped_img_240, cropped_img_0, cv2.TM_CCOEFF_NORMED)

    min_val, max_val_1, min_loc, max_loc = cv2.minMaxLoc(sim_score_1)
    min_val, max_val_2, min_loc, max_loc = cv2.minMaxLoc(sim_score_2)

    print("Similarity scores: ", max_val_1, ", ", max_val_2, "\n\n")
    return (max_val_1, max_val_2)


def below_th(scores):
    below_th = False
    for score in scores:
        if score > 0 and score < similarity_th:
            below_th = True
        if score < 0 and score > (1 - similarity_th):
            below_th = True
    return below_th


def draw_line(img, box, theta, rot_rect, color, length, shift=0):
    (x, y), (w, h), ang = rot_rect
    Cx, Cy = int(x + shift), int(y + shift)
    x_1 = int(Cx - length * np.cos(theta))
    y_1 = int(Cy - length * np.sin(theta))
    x_2 = int(Cx + length * np.cos(theta))
    y_2 = int(Cy + length * np.sin(theta))
    cv2.line(img, (x_1, y_1), (x_2, y_2), color, 4)

    return img


def draw_star_6(img, box, rot_rect):
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = box
    ref_angle = (y2 - y1) / (x2 - x1)

    theta1 = ref_angle + np.pi / 6
    theta2 = ref_angle + (np.pi / 6) * 2
    theta3 = ref_angle + (np.pi / 6) * 3
    theta4 = ref_angle + (np.pi / 6) * 4
    theta5 = ref_angle + (np.pi / 6) * 5

    #     img = draw_line(img,box,ref_angle, rot_rect, 255, int(abs(x2-x1)/2))
    img = draw_line(img, box, ref_angle, rot_rect, 255, int((x2 - x1) / 2))
    img = draw_line(img, box, theta1, rot_rect, 255, int((x2 - x1) / 2))
    img = draw_line(img, box, theta2, rot_rect, 255, int((x2 - x1) / 2))
    img = draw_line(img, box, theta3, rot_rect, 255, int((x2 - x1) / 2))
    img = draw_line(img, box, theta4, rot_rect, 255, int((x2 - x1) / 2))
    img = draw_line(img, box, theta5, rot_rect, 255, int((x2 - x1) / 2))

    #     cv2.putText(img, "*6", ( (int((x1+x2)/2)+(int((x1+x2)/8))), (int((y1+y2)/2)+30) ), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2, cv2.LINE_8)

    return img


def draw_star_3(img, box, rect_lattice_3):
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = box
    (x, y), (w, h), ang = rect_lattice_3

    ref_angle = (y2 - y1) / (x2 - x1)
    #     ref_angle = rot_rect[2]
    #     if (ref_angle < 0):
    #         ref_angle += 180
    theta1 = ref_angle + np.pi / 3
    theta2 = ref_angle + 2 * np.pi / 3

    img = draw_line(img, box, ref_angle, rect_lattice_3, (0, 255, 0), 30, buffer * move_3_rat)
    img = draw_line(img, box, theta1, rect_lattice_3, (0, 255, 0), 30, buffer * move_3_rat)
    img = draw_line(img, box, theta2, rect_lattice_3, (0, 255, 0), 30, buffer * move_3_rat)

    #     cv2.putText(img, "*3", ( int(x)+int(10*move_3_rat),int(y)+40 ), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2, cv2.LINE_8)
    return img


def draw_star_2(img, box, rect_lattice_2):
    (x, y), (w, h), ang = rect_lattice_2
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = box

    ref_angle = (y2 - y1) / (x2 - x1)
    #     ref_angle = rot_rect[2]
    #     if (ref_angle < 0):
    #         ref_angle += 360
    theta1 = ref_angle + np.pi / 2

    img = draw_line(img, box, ref_angle, rect_lattice_2, (255, 0, 255), 30, buffer * move_2_rat)
    img = draw_line(img, box, theta1, rect_lattice_2, (255, 0, 255), 30, buffer * move_2_rat)

    #     cv2.putText(img, "*2", ( int(x)+int(10*move_2_rat),int(y) ), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,255), 2, cv2.LINE_8)
    return img


def summary(img, signature_6, signature_3, signature_2, largestBox, rot_rect, rect_lattice_3, rect_lattice_2):
    plt.figure(figsize=(14, 14))
    if (signature_6 and signature_3 and signature_2):
        final_img = img.copy()
        final_img = draw_star_6(final_img, largestBox, rot_rect)
        final_img = draw_star_3(final_img, largestBox, rect_lattice_3)
        final_img = draw_star_2(final_img, largestBox, rect_lattice_2)

        red_patch = mpatches.Patch(color=('r'), label='*6')
        blue_patch = mpatches.Patch(color='lawngreen', label='*3')
        green_patch = mpatches.Patch(color='fuchsia', label='*2')
        plt.legend(handles=[red_patch, blue_patch, green_patch], fontsize=20)

        # plt.imshow(final_img)
        # plt.title("Symmetry signature is *632", fontsize=25)
        # plt.show()
        #         return True
        return (True, final_img)

    else:
        # plt.imshow(img)
        # plt.title("Symmetry signature is NOT *632", fontsize=25)
        # plt.show()
        #         return False
        return (False, img)


# img_name = 'C:/Users/youva/OneDrive/Desktop/CS-HIT/3rd-year/semester-A/symmetry-and-fractals/test/star632.jpg'
# similarity_th = 0.8
# gaussian_factor = 9
# morphology_kernel = 9
# adapt_th_const = -1
# buffer = 15
# move_2_rat = 0
# move_3_rat = 0

def run(img_name, similarity_th, gaussian_factor, morphology_kernel, adapt_th_const, buffer, move_2_rat, move_3_rat):
    # Init 6 axis reflection indicators
    axis_0_deg = False
    axis_30_deg = False
    axis_60_deg = False
    axis_90_deg = False
    axis_120_deg = False
    axis_150_deg = False

    signature_6 = False
    signature_3 = False
    signature_2 = False

    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

    img_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_bw = cv2.GaussianBlur(img_bw, (gaussian_factor, gaussian_factor), 0)

    img_th = cv2.adaptiveThreshold(img_bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, adapt_th_const)

    kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)
    dilation = cv2.dilate(img_th, kernel)
    closing = cv2.erode(dilation, kernel)

    edges = cv2.Canny(closing, 30, 100, L2gradient=True)

    ctrs, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt_groups = []
    matches = []
    matches_id = []
    used = []
    for i in range(len(ctrs)):
        M = cv2.moments(ctrs[i])
        if (M['m00'] < 200):
            continue
        matches = []
        matches_id = []
        for j in range(len(ctrs)):
            ret = cv2.matchShapes(ctrs[j], ctrs[i], 1, 0.0)
            if (ret < 0.15):
                matches.append(ctrs[j])
                matches_id.append(j)
        if (len(matches) > 2) and (intersection(matches_id, used) == 0):
            r, g, b = np.random.randint(255), np.random.randint(255), np.random.randint(255)
            cnt_groups.append(matches)
            used.append(matches_id)

    boxes = []
    for grp in cnt_groups:
        angles = []
        different = []
        r, g, b = np.random.randint(255), np.random.randint(255), np.random.randint(255)
        for i in range(len(grp)):
            (x, y), (w, h), angle = cv2.minAreaRect(grp[i])
            rot_rect = cv2.minAreaRect(grp[i])
            box = np.int64(cv2.boxPoints(rot_rect))
            angles.append(roundup(angle))
            if [(x, y), roundup(angle)] not in different:
                different.append([(x, y), roundup(angle)])
                boxes.append(box)

    img_copy = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

    maxArea = 0
    largestBox = None
    areas = []
    Cx, Cy = 0, 0

    # Finding tile
    for box in range(len(boxes)):
        M = cv2.moments(boxes[box])
        areas.append(M['m00'])
        if (M['m00']) > maxArea:
            maxArea = M['m00']
            largestBox = boxes[box]
            Cx = int(M['m10'] / M['m00'])
            Cy = int(M['m01'] / M['m00'])

    rot_rect = cv2.minAreaRect(largestBox)
    rot_rect = ((rot_rect[0][0], rot_rect[0][1]), (rot_rect[1][0] + buffer, rot_rect[1][1] + buffer), rot_rect[2])
    largestBox = np.int64(cv2.boxPoints(rot_rect))

    img_copy_for_crop = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    crop_img = crop_minAreaRect(img_copy_for_crop, rot_rect)

    dim_0d = axis_reflective_test(crop_img, 0, rot_rect)
    axis_0_deg = sim_in_th(dim_0d, axis_0_deg, "0 degrees axis")
    dim_30d = axis_reflective_test(crop_img, 30, rot_rect)
    axis_30_deg = sim_in_th(dim_30d, axis_30_deg, "30 degrees axis")
    dim_60d = axis_reflective_test(crop_img, 60, rot_rect)
    axis_60_deg = sim_in_th(dim_60d, axis_60_deg, "60 degrees axis")
    dim_90d = axis_reflective_test(crop_img, 90, rot_rect)
    axis_90_deg = sim_in_th(dim_90d, axis_90_deg, "90 degrees axis")
    dim_120d = axis_reflective_test(crop_img, 120, rot_rect)
    axis_120_deg = sim_in_th(dim_120d, axis_120_deg, "120 degrees axis")
    dim_150d = axis_reflective_test(crop_img, 150, rot_rect)
    axis_150_deg = sim_in_th(dim_150d, axis_150_deg, "150 degrees axis")

    signature_6_axis_lst = [axis_0_deg, axis_30_deg, axis_60_deg, axis_90_deg, axis_120_deg, axis_150_deg]
    if all(signature_6_axis_lst):
        signature_6 = True

    img_copy_for_crop = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    crop_img = crop_minAreaRect(img_copy_for_crop, rot_rect)

    intesections_img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

    (x, y), (w, h), angle = rot_rect
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = largestBox

    r, g, b = np.random.randint(255), np.random.randint(255), np.random.randint(255)
    rot_rect_shift1_org = (((x1 + x4) / 2), ((y1 + y4) / 2)), (w, h), angle
    rot_rect_shift1_padded = (((x1 + x4) / 2), ((y1 + y4) / 2)), (w * 1.5, h * 1.5), angle
    box_shift1_org = np.int64(cv2.boxPoints(rot_rect_shift1_org))
    box_shift1_padded = np.int64(cv2.boxPoints(rot_rect_shift1_padded))
    cv2.drawContours(intesections_img, [box_shift1_org], 0, (r, g, b), 3)
    # cv2.drawContours(intesections_img,[box_shift1_padded], 0, (r,g,b), 2)

    r, g, b = np.random.randint(255), np.random.randint(255), np.random.randint(255)
    rot_rect_shift2_org = (int((x2 + x3) / 2), int((y2 + y3) / 2)), (w, h), angle
    rot_rect_shift2_padded = (int((x2 + x3) / 2), int((y2 + y3) / 2)), (w * 1.5, h * 1.5), angle
    box_shift2_org = np.int64(cv2.boxPoints(rot_rect_shift2_org))
    box_shift2_padded = np.int64(cv2.boxPoints(rot_rect_shift2_padded))
    cv2.drawContours(intesections_img, [box_shift2_org], 0, (r, g, b), 3)
    # cv2.drawContours(intesections_img,[box_shift2_padded], 0, (r,g,b), 2)

    r, g, b = np.random.randint(255), np.random.randint(255), np.random.randint(255)
    rot_rect_shift3_org = (((x1 + x2) / 2), ((y1 + y2) / 2)), (w, h), angle
    rot_rect_shift3_padded = (((x1 + x2) / 2), ((y1 + y2) / 2)), (w * 1.5, h * 1.5), angle
    box_shift3_org = np.int64(cv2.boxPoints(rot_rect_shift3_org))
    box_shift3_padded = np.int64(cv2.boxPoints(rot_rect_shift3_padded))
    cv2.drawContours(intesections_img, [box_shift3_org], 0, (r, g, b), 3)
    # cv2.drawContours(intesections_img,[box_shift3_padded], 0, (r,g,b), 2)

    r, g, b = np.random.randint(255), np.random.randint(255), np.random.randint(255)
    rot_rect_shift4_org = (int((x3 + x4) / 2), int((y3 + y4) / 2)), (w, h), angle
    rot_rect_shift4_padded = (int((x3 + x4) / 2), int((y3 + y4) / 2)), (w * 1.5, h * 1.5), angle
    box_shift4_org = np.int64(cv2.boxPoints(rot_rect_shift4_org))
    box_shift4_padded = np.int64(cv2.boxPoints(rot_rect_shift4_padded))
    cv2.drawContours(intesections_img, [box_shift4_org], 0, (r, g, b), 3)
    # cv2.drawContours(intesections_img,[box_shift4_padded], 0, (r,g,b), 2)

    intesections_img_crop = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

    crop_img_sig32_11 = crop_minAreaRect(intesections_img_crop, rot_rect_shift1_padded)
    crop_img_sig32_12 = crop_minAreaRect(intesections_img_crop, rot_rect_shift2_padded)
    nonZero11 = cv2.countNonZero(cv2.cvtColor(crop_img_sig32_11, cv2.COLOR_RGB2GRAY))
    zeros_11 = crop_img_sig32_11.shape[0] * crop_img_sig32_12.shape[1] - nonZero11
    nonZero12 = cv2.countNonZero(cv2.cvtColor(crop_img_sig32_12, cv2.COLOR_RGB2GRAY))
    zeros_12 = crop_img_sig32_12.shape[0] * crop_img_sig32_12.shape[1] - nonZero12
    if zeros_11 > zeros_12:
        crop_img_sig32_1 = crop_img_sig32_12
        crop_img_sig32_1_r = crop_minAreaRect(intesections_img_crop, rot_rect_shift2_org)
        rot_rect_lattice_1 = rot_rect_shift2_org
    else:
        crop_img_sig32_1 = crop_img_sig32_11
        crop_img_sig32_1_r = crop_minAreaRect(intesections_img_crop, rot_rect_shift1_org)
        rot_rect_lattice_1 = rot_rect_shift1_org

    crop_img_sig32_31 = crop_minAreaRect(intesections_img_crop, rot_rect_shift3_padded)
    crop_img_sig32_32 = crop_minAreaRect(intesections_img_crop, rot_rect_shift4_padded)
    nonZero31 = cv2.countNonZero(cv2.cvtColor(crop_img_sig32_31, cv2.COLOR_RGB2GRAY))
    zeros_31 = crop_img_sig32_31.shape[0] * crop_img_sig32_32.shape[1] - nonZero31
    nonZero32 = cv2.countNonZero(cv2.cvtColor(crop_img_sig32_32, cv2.COLOR_RGB2GRAY))
    zeros_32 = crop_img_sig32_32.shape[0] * crop_img_sig32_32.shape[1] - nonZero32
    if zeros_31 > zeros_32:
        crop_img_sig32_2 = crop_img_sig32_32
        crop_img_sig32_2_r = crop_minAreaRect(intesections_img_crop, rot_rect_shift4_org)
        rot_rect_lattice_2 = rot_rect_shift4_org
    else:
        crop_img_sig32_2 = crop_img_sig32_31
        crop_img_sig32_2_r = crop_minAreaRect(intesections_img_crop, rot_rect_shift3_org)
        rot_rect_lattice_2 = rot_rect_shift3_org

    res1, res2, res3, res4 = axis_reflective_test(crop_img_sig32_1_r, 90, rot_rect), axis_reflective_test(
        crop_img_sig32_2_r, 90, rot_rect), axis_reflective_test(crop_img_sig32_1_r, 0, rot_rect), axis_reflective_test(
        crop_img_sig32_2_r, 0, rot_rect)
    signature_2 = sim_in_th(res1, signature_2, "lattice 1 90 degrees") or sim_in_th(res2, signature_2,"lattice 2 90 degrees") or sim_in_th(res3, signature_2, "lattice 1 0 degrees") or sim_in_th(res3, signature_2, "lattice 2 0 degrees")

    ret_lattice1 = check_3_rotations(1, crop_img_sig32_1)
    ret_lattice2 = check_3_rotations(2, crop_img_sig32_2)

    rect_lattice_3 = None
    rect_lattice_2 = None
    if (below_th(ret_lattice1) == False):
        signature_3 = True
        rect_lattice_3 = rot_rect_lattice_1
        rect_lattice_2 = rot_rect_lattice_2
    elif (below_th(ret_lattice2) == False):
        signature_3 = True
        rect_lattice_3 = rot_rect_lattice_2
        rect_lattice_2 = rot_rect_lattice_1

    #     return [img, signature_6, signature_3, signature_2, largestBox, rot_rect, rect_lattice_3, rect_lattice_2]
    if rect_lattice_3 is None:
        return (False, img)
    else:
        res, final = summary(img, signature_6, signature_3, signature_2, largestBox, rot_rect, rect_lattice_3, rect_lattice_2)
        return (res, final)

def main():
    res, final = run()
    title = ""
    if res==True:
        title = "Symmetry signature is *632"
    else:
        title = "Symmetry signature is NOT *632"

    cv2.imshow(title, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))  # works in BGR !!
    cv2.waitKey(0)  # waits for a keystroke (returns its ASCII code)

    cv2.destroyAllWindows()
    cv2.waitKey(1)  # wait 1 msec (=1/1000 of a second)

if __name__ == "__main__":
    main()


def __init__():
    pass