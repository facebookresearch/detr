# Data Curation -> Duplicates

Script to remove duplicate images from a set of directories. The initial aim is to be used after scraping the search string

##### Requiremnets:
python modules: 

	* numpy

	* argparse

	* cv2

	* os


## Installation

no installation, just run with python

## Arguments
-D: Directory holding images to be cleaned. Default = "./fridge"

-M: Minimum size (h or w) below which images should be deleted. Default = 200

## Usage

```python
python handleDuplications.py -D <directory> -M <min_size>

```
## Example
```python
python handleDuplications.py -D ".\fridge_photos" -M 350

```
Duplicate images in ALL the subdirectories of the given directory are going to be deleted
