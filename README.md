# imgscraper for artists from the artist: collecting references - description, features and dependencies

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

<img width="975" height="476" alt="image" src="https://github.com/user-attachments/assets/8b966e21-a61f-4394-8506-ab993ec5819b" />

<img width="976" height="473" alt="image" src="https://github.com/user-attachments/assets/287a014d-3287-4823-b564-0a649a30ba2a" />

<img width="974" height="475" alt="image" src="https://github.com/user-attachments/assets/34adabe1-781b-416c-b1f0-fd04b069570e" />

<img width="974" height="655" alt="image" src="https://github.com/user-attachments/assets/74aa1c0f-cfea-43f7-90f6-a1b9cd32dbd6" />

<img width="983" height="602" alt="image" src="https://github.com/user-attachments/assets/49611aa5-0796-4b0f-9575-c005f8fd849f" />


# imgscraper for artists from the artist: collecting references - setup       

          1. clone the repository
             git clone https://github.com/rambaeharambae/img_drawing_reference_scraper.git
          2. install dependencies
             pip install -r requirements.txt
          3. run via python
             python imgscraper.py
          4. build an .exe executable file
             pyinstaller --onefile --noconsole imgscraper.py

# # imgscraper for artists by artist: collecting references - download
          go to release and download .exe

# # imgscraper for artists by artist: collecting references - knows bugs
          Shutdown Behavior sometimes doesn't work as expected despite PSUTIL:

          Check in the task manager that chrome / chromedriver processes are not left hanging. 
          If everything is correct, they should end shortly after closing the application and webdriver\selenium, but it doesn't.

          So my suggestion while exiting the program is — in Filters tab, press stopping browser and then exit the program.

