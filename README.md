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

<img width="1185" height="591" alt="image" src="https://github.com/user-attachments/assets/c2e6399b-6ddb-4df7-bfe8-ab9a58a5b997" />
<img width="1182" height="600" alt="image" src="https://github.com/user-attachments/assets/24622bbe-906c-46e6-b48b-0135a3635919" />
<img width="1228" height="601" alt="image" src="https://github.com/user-attachments/assets/ec7b705a-2a5d-4286-b225-98e47d68cabb" />
<img width="1226" height="596" alt="image" src="https://github.com/user-attachments/assets/8c77b11c-a7d6-4a60-915f-d524a24b321f" />

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
          If you get too much not what you wanted to parse (heads instead of full body person) — change full-body ratio threshold or use additional requests or both.
          ---
          The first launch could be long if you're missing some dependencies.
