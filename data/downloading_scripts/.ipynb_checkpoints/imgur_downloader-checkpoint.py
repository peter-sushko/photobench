"""
This module provides functionality to download images from Imgur. It supports the following types of Imgur URLs:
1. Direct image links (i.imgur.com)
2. Post links (imgur.com/...)
3. Albums and galleries (imgur.com/a/... or imgur.com/gallery/...)

Functions:
- clasify_url: Classifies an Imgur URL as an image, post, or album/gallery.
- find_image_url_extension: Determines the correct extension for an Imgur image URL.
- download_imgur_image: Downloads an image from a direct Imgur URL.
- download_imgur_post: Converts a post URL to a direct image URL and downloads it.
- get_imgur_album_info: Retrieves image information from an album or gallery URL.
- download_imgur_album: Downloads images from an Imgur album or gallery.
"""

import re
import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import json

folder = "test_download_images"

def clasify_url(imgur_link:str):
    """
    Classifies an Imgur URL based on its type.
    
    Parameters:
        imgur_link (str): The Imgur URL to classify.
        
    Returns:
        str: The type of the Imgur URL ('image', 'album', 'post', or -1 if unclassified).
    """
    if 'i.imgur.com' in imgur_link:
        return 'image'
    elif '/a/' in imgur_link or '/gallery/' in imgur_link:
        return 'album'
    elif 'imgur.com' in imgur_link:
        return 'post'
    return -1

def find_image_url_extension(imgur_url: str) -> str:
    """
    Determines the correct file extension for an Imgur image URL.
    
    Parameters:
        imgur_url (str): The Imgur URL to analyze.
        
    Returns:
        str: The file extension of the image (default is 'jpeg' if not found).
    """
    match = re.search(r'\.([a-zA-Z0-9]+)(?:\?.*)?$', imgur_url)
    if match:
        return match.group(1)
    return 'jpeg' #in case we dont find an extension

def download_imgur_image(url: str, image_id: str, download_folder: str, verbose: bool = True):
    """
    Downloads an image from a direct Imgur URL.
    
    Parameters:
        url (str): The direct Imgur image URL.
        image_id (str): The ID of the image.
        download_folder (str): The folder to save the downloaded image.
        verbose (bool): Whether to print status messages (default is True).
        
    Returns:
        str: The status of the download ('success', 'Failure: ...', or 'Error ...').
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    retry_strategy = Retry(
        total=3,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry
        backoff_factor=2  # Wait 2, 4, 8 seconds between retries
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    status = 'Not Attempted'

    try:
        response = http.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'image' in content_type:
                status = 'success'
                extension = find_image_url_extension(url)
                filename = f"{image_id}.{extension}"  # Use the correct extension
                filepath = os.path.join(download_folder, filename)
                
                with open(filepath, 'wb') as out_file:
                    out_file.write(response.content)
                    
                # Check if we got the placeholder 404 image:
                downloaded_image = Image.open(filepath)
                placeholder_image = Image.open('/Users/petrsushko/Desktop/placeholder.jpg')
                if list(downloaded_image.getdata()) == list(placeholder_image.getdata()):
                    os.remove(filepath)
                    status = 'Failure: Placeholder image downloaded'
                    
            else:
                status = 'Failure: Not an image'
        else:
            status = f"Error {response.status_code}"
    except Exception as e:
        status = 'Failure: ' + str(e)

    if url == 'unrecovered':
        status = 'URL wasnt recovered'
    if verbose:
        print(image_id, status)
    return status

def download_imgur_post(url: str, image_id: str, download_folder: str, verbose: bool = True):
    """
    Converts an Imgur post URL to a direct image URL and downloads the image.
    
    Parameters:
        url (str): The Imgur post URL.
        image_id (str): The ID of the image.
        download_folder (str): The folder to save the downloaded image.
        verbose (bool): Whether to print status messages (default is True).
        
    Returns:
        str: The status of the download attempt ('success', 'failed_to_convert_to_image_url', or other error messages).
    """
    pattern = r'://imgur\.'
    if re.search(pattern, url):
        image_url = re.sub(pattern, '://i.imgur.', url, count=1)
        attempt = download_imgur_image(image_url + '.jpeg', image_id, download_folder, verbose=verbose)
        return attempt            
    return 'failed_to_convert_to_image_url'

def get_imgur_album_info(album_url):
    """
    Retrieves image information from an Imgur album or gallery URL.
    
    Parameters:
        album_url (str): The Imgur album or gallery URL.
        
    Returns:
        str: The direct image URL (or 'unrecovered' if not found).

    Note: would be great if it could also return the number of images in the album.
    I couldn't figure it out.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    image_url = 'unrecovered'
    response = requests.get(album_url, headers=headers)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Get image URL
    meta_tag = soup.find('meta', property='og:image')
    image_url = meta_tag.get('content', None).split('?')[0]
    if str(image_url)=='':
        image_url = 'unrecovered'

    return image_url

def download_imgur_album(url: str, image_id: str, download_folder: str, verbose: bool = True):
    """
    Downloads images from an Imgur album or gallery.
    
    Parameters:
        url (str): The Imgur album or gallery URL.
        image_id (str): The ID of the album.
        download_folder (str): The folder to save the downloaded images.
        verbose (bool): Whether to print status messages (default is True).
        
    Returns:
        str: The status of the download ('success', 'Failure: ...', or 'Error ...').
    """
    url_to_download = get_imgur_album_info(url)
    attempt = download_imgur_image(url_to_download , image_id, download_folder, verbose=verbose)
    return attempt 