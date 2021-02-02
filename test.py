from time import time

import matplotlib.pyplot as plt
from momentum import main, get_spline

if __name__ == "__main__":
    #img = plt.imread("border.png")#[..., 0]
    #img = plt.imread("0019.png")#[..., 0]
    #img = plt.imread("img_err_0.png")[..., 0]
    img = plt.imread("test_v3.png")
    # Get spline coordinates
    t0 = time()
    x, y, img_skeleton = main(img)
    t1 = time()
    print("TIME:", t1 - t0)

    # Convert spline coordinates to image frame
    spline_frame = get_spline(x=y, y=x, width=640, height=480)

    # Draw outputs
    plt.subplot(121)
    plt.imshow(img_skeleton)
    plt.subplot(122)
    plt.imshow(spline_frame)
    plt.show()
