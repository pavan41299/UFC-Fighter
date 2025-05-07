Run Command : 
py -3.9 -m streamlit run main.py
(make sure to select a browser with JavaScript enabled- Chrome or Edge works without any additional things)


Requirements:
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.22.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
tensorflow>=2.11.0
uuid 
(built-in in Python, no need to install separately)


Video implementation link :
https://drive.google.com/file/d/1f_IEbDTOf57xONtjY-5MFHNuZLRW_J9L/view?usp=sharing


File Structure:

1. pro.py : We are using it to find the round-wise iterations about the fighter and its analytics
2. main.py : Here we are running all the models to perform the train, test and others.
3. scrapper.py: We are using this to scrape the information from the website
4. smal_store: trained weights of the data
5. csv files are the data used

