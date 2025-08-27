# Genetic Genealogy Path Analyzer

A Flask-based web application designed for genetic genealogists to automatically find, map, and visualize the precise genealogical connection between a root person and their DNA matches.

This tool bridges the gap between a documented family tree (a GEDCOM file) and a list of DNA relatives (a CSV file exported from services like GEDmatch). It identifies individuals present in both files and displays the step-by-step ancestral path, a relationship prediction based on shared DNA, and an interactive graph of the connection.

## Features

-   **Web-Based Interface:** Easy-to-use interface built with Flask. No command-line knowledge is needed to use the tool.
-   **File Uploads:** Accepts a standard **GEDCOM (.ged)** file for the family tree and a **CSV (.csv)** file for the DNA match list.
-   **Automatic Match Identification:** Uses fuzzy name matching to intelligently find individuals who appear in both your family tree and your DNA match list, accounting for minor spelling differences.
-   **DNA Data Aggregation:** Correctly processes segment-based CSV files (like those from GEDmatch's Ancestor Project) by grouping all segments for a single person and summing the shared centiMorgans (cM) to get an accurate total.
-   **Shortest Path Calculation:** For each confirmed match, it uses the NetworkX library to calculate the shortest genealogical path from you to that person.
-   **Relationship Prediction:** Translates the total shared cM value into a list of probable relationships (e.g., "2nd Cousin," "1st Cousin once removed") based on data from the Shared cM Project.
-   **Interactive Graph Visualization:** Generates a clean, interactive, and hierarchical graph for each connection using Pyvis, making it easy to visualize the ancestral path.
-   **Sorted Results:** The final results are sorted by the highest shared cM value, showing the most significant matches first.

## How It Works

1.  The user uploads their GEDCOM and CSV files via the web form and provides their name as the starting point.
2.  The application parses the GEDCOM file into a `networkx` graph structure, modeling individuals and families.
3.  It then reads the DNA match CSV file with `pandas`, automatically cleaning column headers and aggregating segment data to get a total cM value for each unique match.
4.  The core logic iterates through the aggregated DNA matches, using `thefuzz` to find a corresponding person in the GEDCOM graph.
5.  When a match is found, `networkx.shortest_path` calculates the genealogical connection.
6.  This path is then rendered as an interactive `pyvis` graph, and the cM value is used to predict the relationship.
7.  All results are collected, sorted, and displayed on the web page.

## Technology Stack

-   **Backend:** Python 3, Flask
-   **Data Processing:** Pandas, NetworkX
-   **Genealogy Parsing:** Ged4py
-   **Fuzzy String Matching:** TheFuzz
-   **Graph Visualization:** Pyvis
-   **Frontend:** HTML, Bootstrap 5

## Getting Started

To run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```

5.  Open your web browser and navigate to `http://127.0.0.1:5000`.