# BankMarketingSegmentation

![BankMarketingSegmentation Banner](assets/BMS Banner.png)

## Project Overview

BankMarketingSegmentation is a data-driven project focused on analyzing and predicting customer behavior in the context of direct marketing campaigns by a Portuguese banking institution. The primary goal of this project is to utilize machine learning techniques to predict whether a client will subscribe to a term deposit, thereby aiding in the optimization of marketing strategies.

## Data Set Information

The dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. These marketing campaigns were based on phone calls, often requiring more than one contact to the same client to assess if the bank product (term deposit) would be subscribed. The dataset provides a rich set of features including bank client data, attributes related to the last contact in the current campaign, and social and economic context attributes.

### Attributes Overview

**Bank Client Data:**
- Age
- Job
- Marital Status
- Education
- Default Credit Status
- Housing Loan Status
- Personal Loan Status

**Related to the Last Contact:**
- Contact Communication Type
- Last Contact Month
- Last Contact Day of the Week
- Last Contact Duration

**Other Attributes:**
- Number of Contacts Performed (Campaign)
- Days Passed Since Last Contact from a Previous Campaign (Pdays)
- Number of Contacts Performed Before this Campaign
- Outcome of the Previous Marketing Campaign (Poutcome)

**Social and Economic Context Attributes:**
- Employment Variation Rate (Emp.var.rate)
- Consumer Price Index (Cons.price.idx)
- Consumer Confidence Index (Cons.conf.idx)
- Euribor 3 Month Rate (Euribor3m)
- Number of Employees (Nr.employed)

### Target Variable
- `y`: Has the client subscribed to a term deposit? (Binary: 'yes', 'no')

## Project Structure

Project/
│
├── src/                   # Source code for the project
│   ├── analysis/          # Scripts for data analysis
│   ├── data/              # Data processing and loading scripts
│   ├── models/            # Machine learning models
│   └── utils/             # Utility scripts
│
├── notebooks/             # Jupyter notebooks for exploration and presentations
├── config/                # Configuration files
├── Dockerfile             # Dockerfile for containerizing the application
├── requirements.txt       # Project dependencies
├── main.py                # Main script to run the project
└── README.md              # Project README


## Installation

Instructions for setting up the project environment.

```bash
# Clone the repository
git clone https://github.com/your-username/BankMarketingSegmentation.git
cd BankMarketingSegmentation

# Install dependencies
pip install -r requirements.txt
```

## Usage

Steps to run the project, scripts, or examples.

```python main.py
```

## Contributing

Contributions to the BankMarketingSegmentation project are welcome!

Fork the repository.
Create a new branch (git checkout -b feature/YourFeature).
Make your changes.
Commit your changes (git commit -am 'Add some feature').
Push to the branch (git push origin feature/YourFeature).
Create a new Pull Request.

## Contact
For any queries or suggestions, feel free to reach out to Amitesh Tripathi.
E-mail: theamiteshtripathi@gmail.com
LinkedIn: https://www.linkedin.com/in/theamiteshtripathi/

## License
This project is licensed under the MIT License - see the LICENSE file for details.

