import argparse
from io import BytesIO

import flickrapi
import urllib3
from PIL import Image

# Flickr api access key
flickr = flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)


def flicker_download(keyword, path, size, amount):
    photos = flickr.walk(text=keyword, tag_mode='all', tags=keyword,
                         extras='url_c', per_page=100, sort='relevance')
    urls = []

    # Get the photos urls
    for photo in photos:
        url = photo.get('url_c')
        if url:
            urls.append(url)
        if len(urls) > amount:
            break

    # Download image from the urls
    for i, url in enumerate(urls):
        if url:
            http = urllib3.PoolManager()
            r = http.request('GET', url)
            _image = Image.open(BytesIO(r.data))

            # Resize the image and save it
            _image = _image.resize((size, size), Image.ANTIALIAS)
            _image.save('{}/{}{}.jpg'.format(path, keyword, i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', help='keyword to the search', default='city')
    parser.add_argument('--size', help='size of the save images', default=256, type=int)
    parser.add_argument('--amount', help='amount of images to download', default=5, type=int)
    parser.add_argument('--pathOut', help='path to save the frames', default='../flicker_images')
    args = parser.parse_args()
    flicker_download(args.keyword, args.pathOut, args.size, args.amount)
