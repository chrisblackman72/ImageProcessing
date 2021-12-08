import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from scipy.ndimage import correlate as calculate
from PIL import Image


IMGDIR = join(".", "images")

def kernel_conv(img, kernel):
    assert type(kernel) == np.ndarray, "Kernel must be a 'numpy.ndarray' object"
    assert img.ndim == kernel.ndim, f"Cannot convolve kernel (of dimension {kernel.ndim}) with image (of dimension {img.ndim})"
    
    new_img = np.zeros_like(img)
    calculate(img, kernel, output=new_img, mode="constant", cval=0.0)
    return new_img


def rgb2grey(img):
    """Input rgb image as np-array, shape (h, w, channels)
    """
    if img.ndim <= 2:
        return img

    n_channels = img.shape[2]
    if n_channels > 3:
        # strip off alpha channel
        # greyscale shouldn't have alphas also
        img = img[..., :3]
        
    # translate using luminosity-weighted function
    transform = np.array([0.2989, 0.5870, 0.1140])  # R, G, B weightings
    
    # apply transform to image
    grey = np.sum(np.multiply(transform, img), axis = 2)

    return grey


def gauss2d(size=3, spread=0.5, mu=0, sig=1):
    x, y = np.meshgrid(np.linspace(-spread,spread,size), np.linspace(-spread,spread,size))
    dst = np.sqrt(x*x+y*y)
    
    return 1./(2*np.pi*sig**2) * np.exp(-( (dst-mu)**2 / ( 2.0 * sig**2 ) ) )


def blur2d(img, kernel_size=3, kernel_type="mean"):
    """Assume 2d image (i.e. non-RGB)
    
    kernal_size: Dimension of a square kernel
    """
    
    kernel = np.ones((kernel_size, kernel_size))
    
    if kernel_type == "gaussian":
        kernel = gauss2d(size=kernel_size, sig=1)

    # normalise kernel
    kernel = kernel / np.sum(kernel)
    
    new_img = np.zeros_like(img)
    calculate(img, kernel, output=new_img, mode="constant", cval=0.0)
    
    return new_img


def blur(img, kernel_size=3, kernel_type="mean", stddev=1, greyscale=False):
    if greyscale:
        img = rgb2grey(img)

    if kernel_type == "gaussian":
        kernel = gauss2d(size=kernel_size, sig=stddev)
    else:
        kernel = np.ones((kernel_size, kernel_size))

    # normalise kernel
    kernel = kernel / np.sum(kernel)
    
    new_img = np.zeros_like(img)
    if greyscale:
        calculate(img, kernel, output=new_img, mode="constant", cval=0.0)
    else:
        # split into RGBa, then recombine
        for chan in range(img.shape[2]):
            new_img[..., chan] = calculate(img[..., chan], kernel, mode="constant", cval=0.0)
    
    return new_img


def sobel(img):
    """Apply a sobel operator to an image to highlight edges.
    """
    
    # horizontal gradient kernel
    k_v = np.array([
        [-1, 0, +1],
        [-2, 0, +2],
        [-1, 0, +1]
    ], dtype=np.float64)
    
    # vertical gradient kernel
    k_h = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [+1, +2, +1]
    ], dtype=np.float64)
       
    # Convolve
    G_x = np.zeros_like(img)
    G_y = np.zeros_like(img)
    calculate(img, k_v, output=G_x, mode="constant", cval=0.0)
    calculate(img, k_h, output=G_y, mode="constant", cval=0.0)
    
    # Calculate gradient magnitude, angle
    G = np.sqrt(G_x**2 + G_y**2)
    angles = np.arctan2(G_y, G_x)
    
    # Normalise output
    G = (G-np.min(G))/(np.max(G)-np.min(G))*255
    
    return G, angles


def sc_sobel(img):
    from scipy.ndimage import sobel as scipy_sobel
    G_x = scipy_sobel(img, axis=1, mode="constant", cval=0.0)
    G_y = scipy_sobel(img, axis=0, mode="constant", cval=0.0)
    
    # Calculate gradient magnitude, angle
    G = np.sqrt(G_x**2 + G_y**2)
    angles = np.arctan2(G_y, G_x)

    # Normalise output
    G = (G-np.min(G))/(np.max(G)-np.min(G))*255
    
    return G, angles


if __name__ == "__main__":
    fname = join(IMGDIR, "example-unsplash.jpg")
    # fname = join(IMGDIR, "256-square-image.png")

    img = Image.open(fname)
    np_img = np.asarray(img)
    
    fig, axs = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(15, 8))

    # convert to array
    axs[0].imshow(np_img)
    axs[0].set_title("Original")

    # convert to greyscale
    bw_img = rgb2grey(np_img)
    axs[1].imshow(bw_img, cmap="gray")
    axs[1].set_title("Greyscale")

    # soften with gaussian blur
    soft_img = blur(bw_img, greyscale=True, kernel_type="Gaussian", stddev=0.5)
    axs[2].imshow(soft_img, cmap="gray")
    axs[2].set_title("Gaussian blur")

    # sobel edge detection
    edges, angles = sobel(soft_img)
    axs[3].imshow(edges, cmap="gray")
    axs[3].set_title("User-Defined Sobel edge detection")

    scipysobel, scipyangles = sc_sobel(soft_img)
    axs[4].imshow(scipysobel, cmap="gray")
    axs[4].set_title("Scipy-Defined Sobel edge detection")

    fig.tight_layout()

    # compare output from both
    diff = edges - scipysobel
    print("Average difference between each user- and scipy-defined Sobel output pixel is ", np.mean(diff[diff != 0]))

    # plot angles
    print("Angles range from ", np.min(angles)/np.pi, "pi to ", np.max(angles)/np.pi, "pi")

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    ang = axs[0].imshow(angles/np.pi)
    axs[0].set_title("User-defined angles (/$\pi$) from Sobel")

    ang = axs[1].imshow(scipyangles/np.pi)
    axs[1].set_title("Scipy-defined angles (/$\pi$) from Sobel")

    fig.tight_layout()
    plt.colorbar(ang, ax=fig.get_axes())

    # im = Image.fromarray(edges.astype(np.uint8)).save("sobel-edge-detect.png")

    # horizontal gradient kernel
    k_1 = np.array([
        [-1./4., 0,  1./4.],
        [   0,   0,    0  ],
        [ 1./4., 0, -1./4.],
    ], dtype=np.float64)
    
    # vertical gradient kernel
    k_2 = np.array([
        [-1, -1, -1],
        [-1,  4, -1],
        [-1, -1, -1]
    ], dtype=np.float64)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(kernel_conv(soft_img, k_1), cmap="gray")
    axs[1].imshow(kernel_conv(soft_img, k_2), cmap="gray")
    fig.tight_layout()
    
    plt.show()