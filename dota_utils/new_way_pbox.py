def poly2pbox(poly):
    # poly = rbox2poly(rbox)#, img_height, img_width)
    # poly[6], poly[7] = poly[0] + poly[4] -poly[2], poly[1] + poly[5] -poly[3]
    pt1, pt2, pt3, pt4 = np.array(poly[0:2]), np.array(poly[2:4]), np.array(poly[4:6]), np.array(poly[6:])
    center = np.array([0.25 * sum(poly[0::2]), 0.25 * sum(poly[1::2])])
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    # distance_square1, distance_square2 = abs(np.divide(pt1 - pt3)), abs(np.divide(pt2 - pt4))
    # print(center)
    if np.divide(abs(np.add(pt1, pt3)  - np.multiply(center, 2)), edge1).all() < 0.02 \
            and np.divide(abs(np.add(pt2, pt4) - np.multiply(center, 2)), edge2).all() < 0.02:
        if abs(edge1 - edge2)/(edge1 + edge2) < 0.02:#0.5:
            # pass
            print(abs(edge1 - edge2))
            # print('distance_square1 == distance_square2')
        else:
            pass
            # print(abs(edge1 - edge2))
            # print('diamond')
        pass # be careful!!! -> distance_square1 != distance_square2, diamond
    else:
        pass
        # print(np.divide(abs(np.add(pt1, pt3)  - np.multiply(center, 2)), distance_square1), \
        #       np.divide(abs(np.add(pt2, pt4) - np.multiply(center, 2)), distance_square2))
        # print('1111')
    xmin, ymin, xmax, ymax = min(poly[0::2]), min(poly[1::2]), \
                             max(poly[0::2]), max(poly[1::2])
    # xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
    #                          max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
    #                          min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
    #                          max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    high ,width = ymax - ymin, xmax - xmin
    # pt1, pt2, pt3, pt4 = poly[0:5]
    # pt1, pt2, pt3, pt4 = poly[0:2], poly[2:4], poly[4:6], poly[6:]
    # print(pt1, pt2, pt3, pt4)

    x_list = [pt1[0], pt2[0], pt3[0], pt4[0]]
    y_list = [pt1[1], pt2[1], pt3[1], pt4[1]]
    arr_x = np.array(x_list)
    arr_y = np.array(y_list)
    left_plot_y = min(arr_y[np.where(arr_x == xmin)])
    left_plot_x = max(arr_x[np.where(arr_y == ymin)])
    # print(arr_y[np.where(arr_x == xmin)], arr_x[np.where(arr_y == ymin)])
    # left_plot_y = y_list[x_list.index(xmin)]
    # left_plot_x = x_list[y_list.index(ymin)]
    alpha = (left_plot_y - ymin) / high
    beta = (left_plot_x - xmin) / width
    if max(alpha, beta) < 0.5 or min(alpha, beta) > 0.5:
        thin_flag = True
    else:
        thin_flag = False
    distance_square1 = width * width + high * high * (1 - 2 * alpha) * (1 - 2 * alpha)
    # distance_square2 = high * high + width * width * (1 - 2 * beta) * (1 - 2 * beta)
    beta_get1 = (1 - math.sqrt(max(distance_square1 - high * high, 0) / (width * width))) / 2
    beta_get2 = (1 + math.sqrt(max(distance_square1 - high * high, 0) / (width * width))) / 2
    if (thin_flag and alpha<0.5) or (not thin_flag and alpha>0.5):
        beta_get = min(beta_get1, beta_get2)
    else:
        beta_get = max(beta_get1, beta_get2)
    # # beta_get = 0
    # print(alpha, beta, beta_get, distance_square1, distance_square2)
    area = math.sqrt(high * high * alpha * alpha + width * width * beta_get * beta_get) * \
           math.sqrt(high * high * (1 - alpha) * (1 - alpha) + width * width * (1 - beta_get) * (1 - beta_get))
    return [xmin, ymin, width, high, alpha, thin_flag], area
