from io import BytesIO

import flickrapi
import urllib3
from PIL import Image

# Flickr api access key
flickr = flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)


def flicker_download(keyword, path='../flicker_images/', resize_shape=(256, 256), amount=5):
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
            _image = _image.resize(resize_shape, Image.ANTIALIAS)
            _image.save('{}{}.jpg'.format(path, i))


flicker_download('city')
