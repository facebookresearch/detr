import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import sys
import argparse
import requests
from PIL import Image
import io
import hashlib

# def configure_logging():
# 	logger = logging.getLogger()
# 	logger.setLevel(logging.DEBUG)
# 	handler = logging.StreamHandler()
# 	handler.setFormatter(
# 		logging.Formatter('[%(asctime)s %(levelname)s %(module)s')
# 	)
# 	logger.addHandler(handler)
# 	return logger

# function to scroll to the end
def scroll_to_end(wd, sleep_between_interactions):
	wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
	time.sleep(sleep_between_interactions)

# generating query
def get_query_url(query):
	search_url = "https://www.google.co.in/search?q={q}&source=lnms&tbm=isch"
	return  search_url.format(q = "+".join(query.rstrip().split()))


def extractImagesURLS(args, web_driver, verbose):
	target_url = get_query_url(args.query) if args.url is None else args.url
	# initializing the web driver
	web_driver.get(target_url)
	image_urls = set()
	image_count = 0
	starting_index = 0
	index = 0
	while image_count < args.number:
		# scrolling til the end to load images
		scroll_to_end(web_driver, args.wait_time + 0.7)
		# finding all the thumbnails images which will be opened later to download original image
		thumbnail_results = web_driver.find_elements_by_css_selector("img.Q4LuWd")
		final_index = len(thumbnail_results)
		if final_index == starting_index:
			print(f"Can't load more images, total images found till now : {len(thumbnail_results)}\n\
				Stopping Searching for more images")
			break
		print(f"Found: {final_index} search results. Extracting links from {starting_index} : {final_index}")
		for img in thumbnail_results[starting_index: final_index]:
			try:
				# clicking the thumbnail to get the original image
				img.click()
				time.sleep(args.wait_time)
			except Exception:
				continue
			# scraping original image
			actual_images = web_driver.find_elements_by_css_selector('img.n3VNCb')
			for actual_image in actual_images:
				if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
					# adding url to the set
					image_urls.add(actual_image.get_attribute('src'))
			image_count = len(image_urls)
			if len(image_urls) >= args.number:
				break
		starting_index = final_index
		print(f'checking for the limit of the number of images to be downloaded: {image_count}')
		if len(image_urls) >= args.number:
			print(f"Found: {len(image_urls)} image links")
			break
		else :
			if verbose : print(f"looking for more...\nSo far, got links for {len(image_urls)}")
			# clicking `load more button` at the end of the google image search web page to load more
			load_more_button = web_driver.find_element_by_css_selector("input.mye4qd")
			if load_more_button:
				web_driver.execute_script("document.querySelector('input.mye4qd').click();")
	web_driver.quit()
	return image_urls

# download images from corresponding to the url
def download_image(url, output_path, verbose):
	try:
		response = requests.get(url)
		image_content = response.content
	except Exception as e:
		print(f"Error - could not download {url} - {e}")
		return None
	try:
		image_file = io.BytesIO(image_content)
		image = Image.open(image_file).convert('RGB')
		file_name = hashlib.sha1(image_content).hexdigest()[:10] + '.jpg'
		file_path = os.path.join(os.path.abspath(output_path), file_name)
		if os.path.isfile(file_path):
			if verbose: print(f"File already exists in the output directory: {file_path}")
		else: 
			with open(file_path, 'wb') as writeFile:
				image.save(writeFile, "JPEG", quality=85)
			if verbose : print(f"image downloaded succesfully {url} - as file {file_path}")
		return file_name
	except Exception as e:
		print(f"Error : could not process downloaded image content\n--> {url} - {e}")
		return None

# download images from urls and saving them in output directory
def fetch_images_from_urls(urls, args, verbose=False):
	url_2_file_name = {}
	print('Starting to download images from fetched urls...')
	count = 0
	old_urls = set()
	if os.path.isfile(args.urls_path):
		with open(args.urls_path, 'r') as readFile:
			for line in readFile:
				old_urls.add(line.rstrip().split('\t')[1])
	for url in urls:
		count += 1
		if url in old_urls:
			print('Image with the current url is already downloaded last time')
			continue
		file_name = download_image(url, args.directory, verbose)
		if file_name is not None:
			url_2_file_name[url] = file_name
		if count % 50 == 0:
			print(f'Total images downloaded from Google Images : {count}/{len(urls)}')
	print("Downloading complete.... generated Url2FileName Dictionary ... :)")
	return url_2_file_name

# writing urls in an tsv file, [just a precautionary measure(not anymore I think)] later help in data manipulation
def writeURLS(url_2_file_name, urls_path):
	print('Starting updating URL - FILE NAME tsv file...')
	url_2_file_name_buffer = {}
	if os.path.isfile(urls_path):
		with open(urls_path, 'r') as readFile:
			for line in readFile:
				# print(line)
				file_name, url = line.rstrip().split('\t')
				url_2_file_name_buffer[url] = file_name
	for url in url_2_file_name:
		url_2_file_name_buffer[url] = url_2_file_name[url]
	with open(urls_path, 'w') as writeFile:
		for url in url_2_file_name_buffer:
			writeFile.write(url_2_file_name_buffer[url] + '\t' + url + '\n')
	print(f'Updating urls finished with the file path : {os.path.abspath(urls_path)}')

def main(args):
	# browser= webdriver.Chrome(<PATH FOR THE CHROME DRIVER>)
	browser= webdriver.Chrome('./chromedriver')
	urls = extractImagesURLS(args, browser, args.verbose)
	url_2_file_name = fetch_images_from_urls(urls, args)
	writeURLS(url_2_file_name, args.urls_path)
	print('Finished... :)')

def argument_parser():
	parser = argparse.ArgumentParser(description='Scrape Google Images')
	parser.add_argument('-Q', '--query', default=None, type=str, help='search query for google images')
	parser.add_argument('-N', '--number', default=100, type=int, help='Total number of images to be downloaded')
	parser.add_argument('-D', '--directory', default='./fridge', type=str, help='directory for storing results')
	parser.add_argument('-CD', '--create_directory', default=False, action='store_true', help='Permission to create directory')
	parser.add_argument('-URL', '--url', type=str, default=None, help='URL for scraping images')
	parser.add_argument('-WT', '--wait_time', type=float, default=0.3, help='wait time between scroll loadings')
	parser.add_argument('-URLP', '--urls_path', default='./scrapped_images_urls_names.tsv', type=str, help='path for writing fetched urls')
	parser.add_argument('-V', '--verbose', action='store_true', help='Enable displaying processing logs')
	args = parser.parse_args()
	if args.query is None and args.url is None:
		raise Exception('Not found any query or any URL to be scraped, ensure that one of them has been passed as argument')
	if not os.path.isdir(os.path.abspath(args.directory)):
		if not args.create_directory:
			raise Exception("Given directory : {}\n doesn\'t exists, pass \'--create_directory\' flag or a valid directory".format(
				os.path.abspath(args.directory)))
		else:
			os.mkdir(os.path.abspath(args.directory))
			print(f'directory created : {os.path.abspath(args.directory)}')
	args.directory = os.path.abspath(args.directory)
	return args

if __name__ == '__main__':
	main(argument_parser())