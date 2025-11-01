import streamlit as st
import requests

st.title("🧠 Google Fact Check Explorer")

st.write("This tool checks your news headline or claim against verified fact-checking sources using Google Fact Check Tools API.")

query = st.text_input("Enter a news headline or claim:")

API_KEY = "AIzaSyCkhaQdU4Mla2mHHrxh3czlse6oPxsfImM"  

if st.button("🔍 Check Fact"):
    if not query.strip():
        st.warning("Please enter a news headline or statement.")
    else:
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={API_KEY}"

        with st.spinner("Fetching fact-check results..."):
            response = requests.get(url)
            data = response.json()

        if "claims" in data:
            st.success(f"✅ Found {len(data['claims'])} related fact-checks.")
            for claim in data["claims"]:
                st.markdown("---")
                st.subheader(claim["text"])
                for review in claim["claimReview"]:
                    st.write(f"**Source:** {review['publisher']['name']}")
                    st.write(f"**Rating:** {review['textualRating']}")
                    st.write(f"[Read Full Article]({review['url']})")
        else:
            st.error("No fact-check results found for this query.")
