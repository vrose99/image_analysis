import cv2
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np
import requests
import io
import PIL.Image
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import urllib.parse
import matplotlib
matplotlib.use('SVG')

def load_and_preprocess_image():
    # Load image and preprocess
    airballoon_path = "https://raw.githubusercontent.com/vrose99/image_processing_final/84a009197582e1b728d42896ed61070f27de6059/imagingApp/staticfiles/images/airballoons.jpg"
    response = requests.get(airballoon_path)
    image_bytes = io.BytesIO(response.content)
    img = PIL.Image.open(image_bytes)
    img = np.array(img)[:, :, ::-1].astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def create_mask(freq, overlay):
    mask = np.zeros((freq.shape[0], freq.shape[1], 4))
    opacity_levels = (overlay - np.min(overlay)) / (np.max(overlay) - np.min(overlay))
    mask[:, :, 0] = 1
    mask[:, :, 3] = 1 - opacity_levels
    return mask

def encode_to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri

def freq_img(overlay=None):
    # Load and preprocess the image
    img = load_and_preprocess_image()

    freq = np.fft.fftshift(fftpack.fft2(img))
    plt.figure(figsize=(8, 3))

    # Display original image
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])

    # Display frequency domain
    plt.subplot(122)
    freq_domain = 20 * np.log(np.abs(freq))
    if np.any(overlay):
        freq_domain = 20 * np.log(np.abs(freq))
        plt.imshow(freq_domain, cmap='gray')
        mask = create_mask(freq, overlay)
        plt.imshow(mask, alpha=0.4)
    else:
        plt.imshow(freq_domain, cmap='gray')
    plt.title('Frequency')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    return encode_to_uri(plt.gcf())

def create_butterworth_highpass_filter(img, cutoff, order):
    (cols, rows) = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    D = np.sqrt((np.arange(rows) - crow) ** 2 + (np.arange(cols) - ccol)[:, np.newaxis] ** 2)
    butterworth_hp = 1 / (1 + (cutoff / D) ** (2 * order))
    return butterworth_hp

def normalize_and_display(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image * 255).astype(np.uint8)

    plt.figure(figsize=(8, 3))
    plt.imshow(image, cmap='gray')
    plt.title('Filtered Image')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    return image

def ideal_highpass_filter(cutoff=5, return_filter=False):
    # Load and preprocess the image
    img = load_and_preprocess_image()

    freq = np.fft.fftshift(fftpack.fft2(img))

    # Ideal high-pass filter
    highpass = np.ones(img.shape)
    (h, w) = highpass.shape[:2]
    highpass_filter = cv2.circle(highpass, (w // 2, h // 2), cutoff, (0, 0, 0), -1)
    filtered_freq = np.multiply(highpass_filter, freq)
    filtered_img = fftpack.ifft2(fftpack.ifftshift(filtered_freq)).astype('float32')

    # Normalize and display filtered image
    filtered_img = normalize_and_display(filtered_img)

    if return_filter:
        return encode_to_uri(plt.gcf()), highpass_filter
    else:
        return encode_to_uri(plt.gcf())

def butterworth_highpass_filter(cutoff_frequency=20, order=2, return_filter=False):
    # Load and preprocess the image
    img = load_and_preprocess_image()

    freq = np.fft.fftshift(fftpack.fft2(img))

    # Butterworth high-pass filter
    butterworth_hp = create_butterworth_highpass_filter(img, cutoff_frequency, order)
    filtered_freq = np.multiply(butterworth_hp, freq)
    filtered_img = fftpack.ifft2(fftpack.ifftshift(filtered_freq))

    # Normalize and display filtered image
    filtered_img = normalize_and_display(filtered_img)

    if return_filter:
        return encode_to_uri(plt.gcf()), butterworth_hp
    else:
        return encode_to_uri(plt.gcf())


def ideal_lowpass_filter(cutoff=5, return_filter=False):
    # Load and preprocess the image
    img = load_and_preprocess_image()

    freq = np.fft.fftshift(fftpack.fft2(img))   

    # Ideal low-pass filter
    hp = np.zeros(img.shape)
    (h, w) = hp.shape[:2]
    lowpass = cv2.circle(hp, (w//2 ,h//2), cutoff, (1,1,1), -1)
    lowpass_filter = np.multiply(lowpass, freq)
    filtered_img = fftpack.ifft2(fftpack.ifftshift(lowpass_filter)).astype('float32') ## abs

    # Normalize and display filtered image
    filtered_img = normalize_and_display(filtered_img)

    if return_filter:
        return encode_to_uri(plt.gcf()), lowpass
    else:
        return encode_to_uri(plt.gcf())
    
def create_butterworth_lowpass_filter(img, cutoff, order):
    (cols, rows) = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    
    # Create a Butterworth low-pass filter
    butterworth_lp = np.zeros((rows, cols))
    D = np.sqrt((np.arange(rows) - crow) ** 2 + (np.arange(cols) - ccol)[:, np.newaxis] ** 2)
    butterworth_lp = 1 / (1 + (D / cutoff) ** (2 * order))
    return butterworth_lp

def butterworth_lowpass_filter(cutoff_frequency = 20, order = 2, return_filter=False):
    img = load_and_preprocess_image()

    freq = np.fft.fftshift(fftpack.fft2(img))

    # Apply Butterworth low pass filter
    butterworth_lp = create_butterworth_lowpass_filter(img, cutoff_frequency, order)
    filtered_freq = np.multiply(butterworth_lp, freq)
    filtered_img = fftpack.ifft2(fftpack.ifftshift(filtered_freq)) # abs

    # Normalize the filtered image
    filtered_img = (filtered_img - np.min(filtered_img)) / (np.max(filtered_img) - np.min(filtered_img))
    filtered_img = (filtered_img * 255).astype(np.uint8)
    
    if return_filter:
        return encode_to_uri(plt.gcf()), butterworth_lp
    else:
        return encode_to_uri(plt.gcf())

def ideal_bandpass_filter(cutoff_low=10, cutoff_high=20, return_filter=False):
    img = load_and_preprocess_image()

    freq = np.fft.fftshift(fftpack.fft2(img))
    
    bandpass = np.ones(img.shape, dtype=np.uint8) * 255
    (h, w) = bandpass.shape[:2]
    cv2.circle(bandpass, (w // 2, h // 2), cutoff_high, (0, 0, 0), -1)
    cv2.circle(bandpass, (w // 2, h // 2), cutoff_low, (255, 255, 255), -1)    
    app_filter = np.multiply(bandpass, freq)
    filtered_img = fftpack.ifft2(fftpack.ifftshift(app_filter)).astype('float32') 

    # Normalize and display filtered image
    filtered_img = normalize_and_display(filtered_img)
    if return_filter:
        return encode_to_uri(plt.gcf()), bandpass
    else:
        return encode_to_uri(plt.gcf())

def ideal_notch_filter(radius=30, return_filter=False):
    img = load_and_preprocess_image()

    freq = np.fft.fftshift(fftpack.fft2(img))

    center_row, center_col = freq.shape[0]//2, freq.shape[1]//2
    
    ## ideal notch
    notch = np.ones(img.shape)
    (h, w) = notch.shape[:2]
    for i in range(h):
        for j in range(w):
            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if distance <= radius:
                notch[i, j] = 0
    notch_filter = np.multiply(notch, freq)
    filtered_img = fftpack.ifft2(fftpack.ifftshift(notch_filter)).astype('float32') ## abs

    # Normalize and display filtered image
    filtered_img = normalize_and_display(filtered_img)

    if return_filter:
        return encode_to_uri(plt.gcf()), notch_filter
    else:
        return encode_to_uri(plt.gcf())

def ideal_comb_filter(period=10, return_filter=False):
    img = load_and_preprocess_image()

    freq = np.fft.fftshift(fftpack.fft2(img))
    
    ## ideal comb
    comb = np.ones(img.shape)
    (h, w) = comb.shape[:2]
    for i in range(h):
        for j in range(w):
            if (i + j) % period == 0:
                comb[i, j] = 0
    comb_filter = np.multiply(comb, freq)
    filtered_img = fftpack.ifft2(fftpack.ifftshift(comb_filter)).astype('float32') ## abs

    # Normalize and display filtered image
    filtered_img = normalize_and_display(filtered_img)

    if return_filter:
        return encode_to_uri(plt.gcf()), comb_filter
    else:
        return encode_to_uri(plt.gcf())