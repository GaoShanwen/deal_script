def get_rbox(poly):
    # print(poly)
    # gt_ind = gt_line.split(' ')
    pt1 = (int(float(poly[0])), int(float(poly[1])))
    pt2 = (int(float(poly[2])), int(float(poly[3])))
    pt3 = (int(float(poly[4])), int(float(poly[5])))
    # pt4 = (int(float(gt_ind[6])), int(float(gt_ind[7])))

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    # angle = 0
    if edge1 > edge2:
        width = edge1
        height = edge2
        if pt1[0] - pt2[0] != 0:
            angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    else:#elif edge2 >= edge1:
        width = edge2
        height = edge1
        # print pt2[0], pt3[0]
        if pt2[0] - pt3[0] != 0:
            angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
        else:
            angle = 90.0
    end_width = width
    end_height = height
    if angle < -45.0:
        angle += 180
    if angle <= 0.0:
        angle = 90 + angle
        end_width = height
        end_height = width
    elif angle > 90.0:
        angle = angle -90
        end_width = height
        end_height = width

    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return [x_ctr, y_ctr, end_width, end_height, angle]
