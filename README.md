# imgscraper for artists from the artist: parser - collecting references - description, features and dependencies

          Description:
            The program is designed for automatic parsing primarily references for artists, downloading, sorting and filtering them from:
            - Pinterest (with support for full-size images)
            - Google Images
            - Bing Images

            Does not require third-party APIs, collection occurs through Selenium/webdriver-manager/icrawler locally.

            Main features:
          • Search images by keywords, categories, poses and additional (separate) requests
          • Automatic filtering:
          - Check for a person in the frame (YOLO + you can change the trigger parameter)
          - Detect a full-length person in the image (you can change the trigger parameter)
          - Filter low-quality files (<10 KB)
          - Filter for black and white images (you can combine Poses + Additional requests = "black and white" + Filter black and white)
          - Check and remove duplicates via pHash (you can change the trigger parameter)
          • Work with history: exclude already loaded images (seen_urls.json)
          • Save full-size URLs in `full_urls.log`
          • Load full-res images
          • Parse Pinterest in windowless\headless mode
          • Index folders for faster duplicate checking
          • Real-time work log on the "Debug" tab
          • Save program settings between launches
          • Selecting the minimum image size (wide\height) when parsing
          • Selecting a preset for parsing poses (foreshortening\dynamic action\perspective pack)
          • Selecting the parsing mode for sketches\lines only
          • Full randomization of the search when parsing (helpful when you search for new material through search engines)

            Notes:
            - All files are saved with unique names (date/time + random tail)
            - The history of downloaded URLs can be reset via the menu
            - Could be ported to Linux\macOS but I don't give a shit 
            - The first launch may be long due to the absence of some dependent libraries on the device (like YOLO\WebDriver-manager)
            
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

<img width="2214" height="1107" alt="image" src="https://github.com/user-attachments/assets/138987f3-81e2-4c28-b8f7-220147e492a4" />


# # imgscraper for artists: parser - collecting references - setup       

          1. clone the repository
             git clone https://github.com/rambaeharambae/img_drawing_reference_scraper.git
          2. install dependencies
             pip install -r requirements.txt
          3. run via python
             python imgscraper.py
          4. build an .exe executable file
             pyinstaller --onefile --noconsole imgscraper.py

# # imgscraper for artists: parser - collecting references - download
          go to release and download .exe

# # imgscraper for artists: parser - collecting references - knows bugs
          Shutdown Behavior sometimes doesn't work as expected despite PSUTIL:

          Check in the task manager that chrome / chromedriver processes are not left hanging. 
          If everything is correct, they should end shortly after closing the application and webdriver\selenium, but it doesn't.

          So my suggestion while exiting the program is — in Filters tab, press stopping browser and then exit the program.

          ---
          If you get too much not what you wanted to parse (heads instead of full body person) — change full-body ratio threshold or use additional requests or both.
          ---
          The first launch could be long if you're missing some dependencies.
          ---
          The file size is big as f cuz its --onefile with full dependencies + webdrivermanager - working with optimizations, will be another release soon
