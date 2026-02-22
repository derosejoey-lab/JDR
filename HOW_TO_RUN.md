# How to Run Your Fundamental Stock Scoring Tool

## What This Tool Does

This tool takes your list of ~595 stocks, pulls live financial data for each one,
scores them on fundamental metrics (ROIC, profitability, valuation, etc.), and
produces a ranked list with a **Fundamental Rating from 0 to 100**.

The results are saved as a CSV file that you can open in **Excel or Google Sheets**.

---

## Step-by-Step Setup (One Time Only — About 5 Minutes)

### Step 1: Install Python (if you don't already have it)

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow **"Download Python"** button
3. Run the installer
4. **IMPORTANT:** Check the box at the bottom that says **"Add Python to PATH"** — this is the most common mistake people make
5. Click **"Install Now"**
6. Wait for it to finish, then close the installer

### Step 2: Download This Project

If you haven't already, download this entire project folder (JDR) to your computer.
A good place to put it: **C:\Users\YourName\Desktop\JDR**

### Step 3: Run the Setup

1. Open the JDR folder on your computer
2. **Double-click `setup_windows.bat`**
3. A black window will open and install the required packages
4. When it says "SETUP COMPLETE!", press any key to close it

---

## Running the Scoring Tool (Whenever You Want Fresh Ratings)

1. Open the JDR folder
2. **Double-click `run_scoring.bat`**
3. A black window will show progress as it pulls data for each stock
4. **This takes about 15-30 minutes** for ~595 stocks (it's pulling real-time data)
5. When it's done, you'll see the top 50 and bottom 20 stocks printed on screen
6. A file called **`fundamental_ratings.csv`** will appear in the folder
7. Open that file in Excel or Google Sheets to see all your ratings

---

## Understanding Your Results

The CSV file has these columns:

| Column | What It Means |
|--------|--------------|
| **FUNDAMENTAL RATING (0-100)** | The overall score. Higher = better fundamentals. 80+ is excellent, 60-80 is good, below 40 is weak. |
| **ROIC Score (1-10)** | How efficiently the company uses its money to make profits |
| **Profitability Score (1-10)** | How much profit the company keeps from each dollar of revenue |
| **Balance Sheet Score (1-10)** | How strong the company's financial position is (low debt, lots of cash) |
| **Growth Score (1-10)** | How fast the company is growing revenue and earnings |
| **Valuation Score (1-10)** | How cheap or expensive the stock is relative to its earnings — higher means cheaper/better value |

### Your Weights
- Valuation: **35%** (heaviest — finding good prices is your priority)
- ROIC & Capital Efficiency: **20%**
- Profitability: **20%**
- Balance Sheet Strength: **15%**
- Growth: **10%**

---

## Troubleshooting

**"Python is not recognized"** — You probably didn't check "Add Python to PATH" during installation. Uninstall Python, reinstall it, and make sure to check that box.

**"Module not found" errors** — Run `setup_windows.bat` again.

**Some stocks show "N/A"** — This is normal. Some stocks (especially foreign-listed ones) don't have complete data available through Yahoo Finance. The tool skips what it can't find and scores based on what's available.

**The tool seems stuck** — It saves progress every 50 stocks. If you need to stop it (close the window), the partial results will be saved as `fundamental_ratings_partial.csv`.

---

## Want to Change the Weights?

Open `fundamental_scorer.py` in any text editor (right-click > Open With > Notepad).
Near the top of the file, you'll see:

```
CATEGORY_WEIGHTS = {
    'roic_efficiency': 0.20,
    'profitability':   0.20,
    'balance_sheet':   0.15,
    'growth':          0.10,
    'valuation':       0.35,
}
```

Change the numbers (they must add up to 1.00). Save the file and run again.
