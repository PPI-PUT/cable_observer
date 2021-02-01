import matplotlib.pyplot as plt
from momentum import main, get_spline

if __name__ == "__main__":
    img = plt.imread("test1.png")[..., 0]
    # Get spline coordinates
    x, y, img_skeleton = main(img)

    # Convert spline coordinates to image frame
    spline_frame = get_spline(x=y, y=x, width=640, height=480)

    # Draw outputs
    plt.subplot(121)
    plt.imshow(img_skeleton)
    plt.subplot(122)
    plt.imshow(spline_frame)
    plt.show()
