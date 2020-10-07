## Google Images Web Scraping Module

It automates seraching and scrolling on google images and fetch the urls of the original image that is being modified to display on the google image web page. Then it downloads images correspondings to those urls.
All the urls are saved in a text file just in case if needed.

##### Requiremnets:
python modules: 

	* selenium

	* argparse

	* PIL

	* io

	* hashlib
	
All those packages can be downloaded straight from pip if not installed

### how to use
First off with the web-driver<included one in the repo is for chrome v85>, So accoring to the distribution each version of google chrome or chromium has their own web-driver which can be downloaded from [here](https://sites.google.com/a/chromium.org/chromedriver/getting-started), there are also the setup instructions but you won't need them unless you have changed the default installation folder while installing chrome/chromium

Once you have chrome/chromium installed and downloaded the chrome-web-driver() then we need to pass the file path for the web driver while initializing it in the script. that is on the __line 110__

then running the script, one could use the following arguments:

    -Q <query> : what you want to search. Default is None
	
    -N <max number of images> : number of images that needs to be downloaded. Default is 100
    
    -D <directory> : output directory. Default is './fridge'
    
    -URL <url> : if one does want to use a url of the google image web page instead of searching here with this script. default is None
    
    -WT <time seconds> : wait time while interacting with the pages to let the page load properly first. Default value is 0.3
    
    -CD <-Flag-> : if directory is not present then continue with craeting a new one or stop.
    
    --urls_path<str> : Path to store the fetched urls{just a precautionary measure}, default is the same directory as the script.

##### example: 
```python download_google_images.py -URL "https://www.google.com/search?as_st=y&tbm=isch&hl=en&as_=food+inside+fridge&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=isz:#imgrc=vunOvmVe3MLIyM" -D ./fridge -N 1 --create_directory```

In this case the script will fetch images from the page corresponding to the given URL and download a single image craeting a new directory called fridge in the current working directory

---

and 
```python download_google_images.py -Q "food in the fridge" -D ./fridge -N 1 --create_directory```
 in this case the script will open a google image serach page and search 'food in the fridge' and download a single image saving in the directory same as the last one.


### Note
The advantage of using the url directly is that one can use google advance search and have their own creiteria for filtering images and then can download all the images using the script in a single run.

__I haven't tested it for large number of downloads, So please let me know if it breaks while doing that__
