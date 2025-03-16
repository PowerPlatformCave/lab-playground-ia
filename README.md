## Setup

1. Copy the `.env.example` file to `.env`:
   ```sh
   cp .env.example .env
   ```

2. Fill in the required environment variables in the `.env` file.

3. Create a virtual environment:
   ```sh
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     ```

5. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

6. Run the application:
   ```sh
   python scraper2D.py
   ```