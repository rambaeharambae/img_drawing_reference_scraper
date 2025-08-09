# imgscraper for artists by artist: collecting references for artists - description, features and dependencies

          Description:
            The program is designed for automatic parsing primarily references for artists, downloading and filtering them from:
            - Pinterest (with support for full-size images)
            - Google Images
            - Bing Images

            Does not require third-party APIs, collection occurs through Selenium/webdriver-manager/icrawler locally.

            Main features:
            • Search images by keywords, categories, poses and additional queries
            • Automatic filtering:
            - Check for a person in the frame (YOLO)
            - Detection of a person in full height in a picture
            - Filtering low-quality files (<10 KB) for less thrash
            - Checking and removing duplicates via pHash
            • Working with history: excluding already downloaded images (seen_urls.json)
            • Saving full-size URLs in `full_urls.log`
            • Ability to index folders for faster duplicate checking
            • Live log of work in the "Debug" tab
            • Saving program settings between launches

            Notes:
            - All files are saved with unique names (date/time + random tail)
            - The history of downloaded URLs can be reset via the menu

            Logic developer and tester: Hara
            Developed and wrote the code: ChatGPT

          Dependencies:
            - Python 3.x
            - PyQt5
            - Selenium
            - webdriver-manager
            - icrawler
            - pillow
            - requests
            - beautifulsoup4
            - imagehash
            - ultralitycs

# imgscraper for artists by artist: collecting references for artists - setup       

          1. clone the repository
             git clone https://github.com/rambaeharambae/img_drawing_reference_scraper.git
          2. install dependencies
             pip install -r requirements.txt
          3. run via python
             python imgscraper.py
          4. build an .exe executable file
             pyinstaller --onefile --noconsole imgscraper.py

# # imgscraper for artists by artist: collecting references for artists - download
          go to release and download .exe
