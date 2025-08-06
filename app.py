from openai import OpenAI
import os
import streamlit as st
import re
import requests
import pandas as pd

client = OpenAI(api_key=st.secrets["openai"]["api_key"])

def build_prompt(user_query):
    return f"""
    You are an e-commerce shopping assistant based out of middle east.
    
    Your job:
    1. Detect if the query is about "planning" (like planning a party, picnic, etc.) or "shopping" (explicit buy orders) or "cooking/recipe".
    2. For planning queries, suggest a list of top 5 most relevant items, in order of relevance, that the user might want to buy online to fulfill the task. Be specific.
        - For example, instead of "return gifts", suggest things like "mini chocolates", "puzzle kits", "coloring books" etc.
        - Suggest items that make sense for the occasion and are typically bought online.
        - Only include **one specific item per search step**
    3. For shopping queries, extract item name, quantity as search query and filters like brand, price, rating, etc.
    4. For **cooking/recipe** queries:
       - Identify the **top 5 essential ingredients or products** required for the recipe that a user can buy online.
       - Only suggest **non-perishable, e-commerce-friendly** items — i.e., things that are commonly sold online such as:
         - packaged spices (e.g., garam masala, turmeric, red chili powder)
         - cooking oils and ghee
         - ginger garlic paste
         - cooking cream, sauces, canned or frozen items (if relevant)
         - rice or packaged mixes (e.g., biryani mix, gravy base)
       - **Avoid** suggesting perishable items like fresh vegetables, milk, raw chicken, etc.
       - Think like an online grocery expert. Suggest items a user would likely need but may not already have at home.
       - Only 1 item per search step.
       - Do not give cooking instructions. Only extract shoppable items.

    5. Output your answer in this format:
    
    intent: planning/shopping  
    search_steps:
    - {{q: "item1"}} or  
    - {{q: "item2", filters: {{brand: "XYZ", max_price: "100"}}}}
    
    Think like an e-commerce expert of middle east — only include things users can buy online, strictly relevant to ecommerce. Don’t mention services like booking a restaurant or sending invites.
    
    Examples:
    
    Input: "Help me plan a kids birthday party"
    Output:
    intent: planning
    search_steps:
    - {{q: "birthday balloons"}}
    - {{q: "chocolate cake"}}
    - {{q: "mini chocolates"}}
    - {{q: "party snacks"}}
    - {{q: "colorful paper plates"}}
    
    Input: "Buy 1kg sugar of MDH under 100 aed, and 2kg tur dal from same brand"
    Output:
    intent: shopping
    search_steps:
    - {{q: "1kg sugar", filters: {{brand: "MDH", max_price: "100"}}}}
    - {{q: "2kg tur dal", filters: {{brand: "MDH"}}}}
    
    Input: {user_query}
    Output:
    """


def get_search_plan(user_query):
    prompt = build_prompt(user_query)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_queries(llm_text):
    pattern = r'q:\s*"(.*?)"'
    return re.findall(pattern, llm_text)


def show_product_carousel(df):
    html = '<div style="display: flex; overflow-x: auto; padding: 10px;">'
    for _, row in df.iterrows():
        html += f'''
        <div style="flex: 0 0 auto; text-align: center; margin-right: 20px;">
            <a href="{row['Product URL']}" target="_blank">
                <img src="{row['Image URL']}" width="150" style="border-radius: 8px;">
            </a>
            <div style="font-weight:bold; margin-top:5px;">{row["Name"][:40]}...</div>
            <div>{row["Brand"]}</div>
            <div>AED {row.get("Sale Price (AED)", row.get("Price (AED)", "N/A"))}</div>
            <div>⭐ {row["Rating"]}</div>
        </div>
        '''
    html += '</div>'
    return html  # Return string, not IPython HTML


def fetch_top_products(query, country_code="AE", limit=2, sort_by="popularity", sort_dir="desc"):
    url = "https://api-app.noon.com/_svc/catalog/api/v3/search"

    params = {
        "q": query,
        "country": country_code,
        "limit": limit,
        "page": 1,
        "sort[by]": sort_by,
        "sort[dir]": sort_dir
    }

    headers = {
    "authority" : "api-app.noon.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Referer": "https://www.noon.com/",
    "Origin": "https://www.noon.com",
    "scheme" : "https",
    # "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    # "Accept-Encoding": "gzip, deflate, br, zstd",
    # "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Cookie": 'visitor_id=1377ce41-a7be-4f7f-b968-010a37a1a5f2; _ym_uid=1720677223793794666; _fbp=fb.1.1721023114914.549776813895348518; _scid=5ea4c34e-b791-4c17-9f71-1d9b57c9de4f; GCP_IAP_UID=113324949211713157026; dcae=1; _tt_enable_cookie=1; th_capi_em=5c5a3c21a9b1551f0ec8d02bd7edc8b1167de12e937a99745d3f6f81c1e7a762; _nrtnetid="nav1.public.eyJ1c2VySWQiOiI1ZDYyOTgwZS0yNjNjLTRjNTUtYjA4MC04MmVmODJlOWYyZTAiLCJzaWQiOiI1MDBiYjE4MC04YzhmLTQwMGMtOTk1MS04YmM4NTBhZWZhZGMiLCJ0aWQiOiI0NTU3OWMxYzQ4MGU5N2I4OGJkMTVkZGYyNzFlZDBmNTYzNjAxYjM5NzQwMmU1ODc3ZGQ2YzA1NWQ0ZGM5NGJkIiwiaWF0IjoxNzMwNzI3MzkwfVg2OENkZWUzTHJzVGR3MUpGK1FIRVllSDlDUVZ3L0lYZkR1US9FakdBYTE4ei9NZUFBaHVEUkQrNTF6YWhicTRBME5Hek9tSC9aalVzWUo4dEpxNVV3dU0zN0xqaExoV0xKZFZheWl1YzRBa0x0SDN2K29JN2IzazFwSzI3ejR6RU1wMVUwbWpvNE1lQ1M1Q2wrL3J6YWdKNjdFU05FeXVuZ0RNbndsTlg2MTg2VDkrdVpXbkpDaTcwaVpMV0VBZXg4V0RjTGNGN29CWFhZbDg4TUdhRUovVi9XOElnUENSdy93aU1ZZFhEbDF1ZlE4R2VmOUx2dkt4WVRFdFVUK3kvUU1CekVTMjlVVUhFR0NKOVZTTFFITkFFOE51V1YxQWVvZng1VTdsbWlHR0RFczllWE5JVzlpNXQxRnZHT3B6Rjh4TGNldTZTUG52L2E3dTNIdm5lc2Z6eVF1clVFYis1RGg0SGdzYllESjFxRWhZSGFSd1Y2Rm16Mk9aVTNBVEVzVm1uRWZMUGtPV1NHcE1OdkNaZUJNS1UwN2xBZHRSVXk3N05vdi9acThKOWtxMEgyUHgrK3hNY0JQVzBqZ0tHU0xpbEpDWGtKbGlJUEFoZjhCWHp0bnNmUks2ZUF0OHdVMCs1QmcvK0FnaEd6eVFHWGZaWjdwT2gzZ241V3Bz.MQ=="; _ttp=JZOnddQZGd7KZPXfIQkc20qU1Jr.tt.1; _pin_unauth=dWlkPVpXSXhPVGcwWkRVdFlqWXhZUzAwWmpsakxUZzRaV1l0TlRJNU9EY3lOakpsTnpkaA; supported-image-formats={"avif":true,"webp":true}; _ga_43G6NV0HZY=GS1.1.1739187627.2.0.1739187631.0.0.0; ph_phc_qNKORfyT0LoPjVeJTJ8FfAhCnpzgGBSkZmT27spzR23_posthog=%7B%22distinct_id%22%3A%2201958983-32b6-7dfd-9ecd-39c5316e92b4%22%2C%22%24sesid%22%3A%5B1742539730936%2C%220195b775-f896-7626-87c3-ecbc744cedb3%22%2C1742539716758%5D%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22%24direct%22%2C%22u%22%3A%22https%3A%2F%2Fwww.noon.com%2Fuae-en%2Fprue-puff-sleeve-tiered-dress-blue%2FN49961258V%2Fp%2F%3Futm_source%3DC1000094L%26utm_medium%3Dreferral%22%7D%7D; th_capi_db=ce7653edd037e816d7412eac845cb13026002c2cc3b34129bc8399d5bad6f6a5; __gads=ID=baef72be5c7f55d6:T=1723055350:RT=1745507491:S=ALNI_MbENyLCU-F52CzjsRtvzENDS5-Uqw; __gpi=UID=00000eb8c37ab559:T=1723055350:RT=1745507491:S=ALNI_MYcaevQRQ-SFOEkpmNSYnr-02jK9A; __eoi=ID=bf2f059439d0fed3:T=1744120524:RT=1745507491:S=AA-AfjZPLdv0ydXoS1ofnqnWpxBh; _gcl_gs=2.1.k1$i1747903570$u51244222; ttcsid_CMSCRUBC77U72P15LFN0=1747901962426::l_k9Ymo42jlpk6KM6U6G.2.1747903599280; _ga_MTC4V6QW17=GS2.1.s1747901961$o16$g1$t1747903611$j20$l0$h0$dxK__u9Iaf9AirYYUJsavL3gC-hQii0abgw; _ga=GA1.2.498398072.1720677220; _abck=A8E42B1082197CCBC98A86A6BA70F59B~0~YAAQPvEBF+zpzy2XAQAAk8aYOg4C61SxwI3ZvztQ1xcuW1CeLALk579TjlUIBiWrk5ZILr9tdRSAOK63HOjf9aoQRixzCrUAkCl5fRPj/Fbrfi7lb8p+f4Ps88s598n99BdWrD7ikCb3Y9DkJIOY7vV45LeEHr27H9oCbJ1CL5cX9GEjvASVej9tUT0Nx265GDyL2F57tnFenYx97p8sZVrQyxP7i02Yb1uzmGg4IanvXRvcupD9jpIXLjNApEf1M1mizdW24SATw9Ue6jECOXdXpjlgf9ZgwFQfLLIaKIMuTJJjcvsIib0yS1pm6v24JlLBEV3dsipst/1LyyacfZ3jKSmSGcfDTIUk0BOVOx4dEug3DTk5nRWNRWhc4/SwWa4gr8MyZ7QshaCl830KSG2tQYOWiopgMfOyXSlWWMJLI9VcduEruXIKu54A4KuLkvYL4EZsTK4cOkUSuFWJQSLc4kVLMQFYsBN0YgjC0aEiuwBnu7CSgMeRmbedStETZD3iEVILSR0z4mYf/P3E~-1~-1~-1; _gcl_aw=GCL.1749631128.CjwKCAiAxKy5BhBbEiwAYiW--56APYmycZBFUZsr5uo_0aBYX9dQ7dks53azpAkJOf0IrEhHUz_J6hoCmTYQAvD_BwE; review_lang=xx; x-location-ecom-ae=eyJsYXQiOiAyNTE4NDEzOTQsICJsbmciOiA1NTI2MTUwODAsICJhZGRyZXNzX2NvZGUiOiAiNGQ4YjE3NTdiMDkxMGJjMzBkYTg4MmZkYmQyNGM1YjMiLCAiYXJlYSI6ICJNYXJhc2kgRHIgLSBCdXNpbmVzcyBCYXkgLSBEdWJhaSAtIER1YmFpIn0; th_capi_fn=4422e151fd18e8ff239a9b97e5ea80e26286cc7ce04cef2cfb3ecab63743216d; ttcsid_CJCRUKJC77U5K7SPETSG=1750924595230::HDqe-IJnl42tR9R3zgr-.1.1750924878285; _gcl_au=1.1.649792036.1751896820; _ym_d=1752475306; nloc=en-ae; _sctr=1%7C1754245800000; ZLD887450000000002180avuid=cd3ce55c-a7ef-4f54-bc9c-ea036951ebdf; x-whoami-headers=eyJ4LWxhdCI6IjI1MTg0MTM5NCIsIngtbG5nIjoiNTUyNjE1MDgwIiwieC1hYnkiOiJ7XCJpcGxfZW50cnlwb2ludC5lbmFibGVkXCI6MSxcIndlYl9wbHBfcGRwX3JldmFtcC5lbmFibGVkXCI6MSxcImNhdGVnb3J5X2Jlc3Rfc2VsbGVyLmVuYWJsZWRcIjoxfSIsIngtZWNvbS16b25lY29kZSI6IkFFX0RYQi1TNSIsIngtbm9vbmluc3RhbnQtem9uZWNvZGUiOiJXMDAxMDYzMDdBIiwieC1hYi10ZXN0IjpbNjEsOTQxLDk2MSwxMDMxLDEwODEsMTA5MCwxMTAxLDExNjIsMTIxMSwxMjUxLDEyOTEsMTMwMSwxMzMxLDEzNjIsMTM3MSwxNDEzLDE0MjEsMTQ1MCwxNDcxLDE1MDIsMTU0MSwxNTgwLDE2MjEsMTY1MCwxNzAxLDE3MjEsMTc1MCwxODExXSwieC1yb2NrZXQtem9uZWNvZGUiOiJXMDAwNjg3NjVBIiwieC1yb2NrZXQtZW5hYmxlZCI6dHJ1ZSwieC1pbnRlcm5hbC11c2VyIjp0cnVlLCJ4LWJvcmRlci1lbmFibGVkIjp0cnVlfQ%3D%3D; nguestv2=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJraWQiOiI4ZGNkNTMwMmRlODI0M2VkOWIwMjBhMmVjZmZlZDE0NiIsImlhdCI6MTc1NDM4NTAwNywiZXhwIjoxNzU0Mzg1MzA3fQ.x9nBB_-zRIqpU-fOKgw8AQc6g1weZ8T_j1EsfCStkj4; AKA_A2=A; _etc=qFEqWRV23rXo9QVu; _uetsid=b0340d9070ff11f09f2c57faffb66605; _uetvid=ec7f92403f4911ef9f1ae5159765f4c9; _scid_r=C7pepMNOt5EeF8txHZtXyd5PGi2VUhGJ2INlNg; _clck=u3wzwt%7C2%7Cfy7%7C0%7C1653; _ScCbts=%5B%22334%3Bchrome.2%3A2%3A5%22%2C%22385%3Bchrome.2%3A2%3A5%22%5D; ttcsid_CFED02JC77U7HEM9PC8G=1754388029450::nEvSFYVU6L7wC0-i4lK7.42.1754389961481; ttcsid=1754388029450::xsdRlN-FRnpCNGUiR-qc.44.1754389961481; _clsk=pj9qkj%7C1754389962043%7C22%7C0%7Cz.clarity.ms%2Fcollect; RT="z=1&dm=noon.com&si=a684c35f-702c-4ffc-836c-55ae623b8e10&ss=mdybj9kj&sl=0&tt=0&rl=1"; _natnetidv2=eyJhbGciOiJLTVNSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdGlkIjoiNDU1NzljMWM0ODBlOTdiODhiZDE1ZGRmMjcxZWQwZjU2MzYwMWIzOTc0MDJlNTg3N2RkNmMwNTVkNGRjOTRiZCIsInNpZCI6IjUwMGJiMTgwLThjOGYtNDAwYy05OTUxLThiYzg1MGFlZmFkYyIsInd0aWQiOiI0NmFhNmRmMy0wMDg5LTQxYzItYTJjOS1kN2ZjYWVjYTUyMDEiLCJpYXQiOjE3NTQzOTExOTYsImV4cCI6MTc1NDM5MTQ5Nn0.0W0gxCu8w5KwYHUCVHs-g9BMayZ-RIT-N-Qv1-Hz7e_sYmJBO9Tt95PQRhchLzU92H735NRMtwCFsFe2jlMylY3BkX3KQuvpNrXOBxvPtQMoYHCpG-P9Y3aTj9xoi9cWmJOaubO3YU72g291mVGghWpPMseVzgRCCk6smzbWM0cSx3EAcfogt4VsWHI1z0XvGrFvSa8cdXF60B6ws61OFJY0E0nk0hFfwIpfzB_6uDi70kc4fi5kc88W3m2bS7sgDbxFTajbrzzehx9pQe2wtMKAkGfBHWUXCkcAWTSq1QT4M4V9TrM7IAP1x4OjMTthn4Osw0IezjUppjwmX2SKyil96k1i_A4kuf-EQ2-tpYHWOaqfg7ODJ_1yZcD1AjeQk-SxSVP4p6t1pEjkkF-dSjymKnAudrsTQG_z_kmarjwfB8qvVWmARRkcjyPidPeKaUl0lJQvsdeUFV2P6mUy3BMwowEGVFgyLvsptsGgT4aPlG_PLuXNQChm70AOY-sp',
    "Sec-Ch-Ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
    "Sec-Ch-Ua-Mobile": "?0",
    # "Sec-Ch-Ua-Platform": "macOS",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests" : "1"
    }

    try:
        response = requests.get(url, params=params, headers=headers)

        # st.write(f"🔗 URL: {response.url}")
        # st.write(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            st.error(f"❌ Failed to fetch products. Status code: {response.status_code}")
            return pd.DataFrame()

        if not response.text.strip():
            st.error("❌ Empty response body. API may be blocking deployed server.")
            return pd.DataFrame()

        try:
            data = response.json()
        except Exception as json_err:
            st.error("❌ Response is not valid JSON:")
            st.text(response.text[:1000])  # Show first 1k characters
            return pd.DataFrame()

        products = data.get("hits", [])[:limit]

        if not products:
            st.warning(f"No products returned for query: {query}")
            return pd.DataFrame()

        results = []
        for product in products:
            image_key = product.get("image_key")
            image_url = f"https://f.nooncdn.com/p/{image_key}.jpg?width=800" if image_key else "N/A"

            results.append({
                "SKU": product.get("sku", "N/A"),
                "SKU Config": product.get("sku_config", "N/A"),
                "Name": product.get("name", "N/A"),
                "Brand": product.get("brand", "N/A"),
                "Image URL": image_url,
                "Price (AED)": product.get("price", "N/A"),
                "Sale Price (AED)": product.get("sale_price", "N/A"),
                "Rating": product.get("product_rating", {}).get("value", "N/A"),
                "Product URL": f"https://www.noon.com/uae-en/{product.get('sku', '')}/p/"
            })

        return pd.DataFrame(results)

    except Exception as e:
        st.exception(f"❌ Exception while fetching products for query: {query}\n{e}")
        return pd.DataFrame()


st.set_page_config(page_title="Noon Smart Shopping Assistant", layout="wide")

st.title("🛍️ noon Assistant")
st.markdown("Enter your query — whether it's a **plan**, a **buying task**, or **recipe support**, and we’ll fetch the top picks!")

user_query = st.text_input("💬 What do you need help with?", placeholder="e.g., Help me plan a beach picnic", key="user_query")

if st.button("Generate Search Plan & Show Products") and user_query:
    with st.spinner("Generating search plan using GenAI..."):
        result = get_search_plan(user_query)
        queries = extract_queries(result)
        st.markdown("#### ✨ Detected Search Steps")
        st.code(result, language="yaml")

    queries = extract_queries(result)
    results = []
    
    for i, q in enumerate(queries):
        # st.markdown(f"### 🔍 Step {i+1}: Searching for `{q}`")
        df_item = fetch_top_products(query=q)
        
        if df_item.empty:
            st.warning(f"No results found for: `{q}`")
        else:
            # st.success(f"✅ Found {len(df_item)} items for: `{q}`")
            # st.dataframe(df_item)  # Debug: show intermediate result
            results.append(df_item)

    if results:
        df = pd.concat(results, ignore_index=True)
        st.markdown("#### 🛒 Top Product Recommendations")
        st.components.v1.html(show_product_carousel(df), height=400, scrolling=True)
    else:
        st.warning("No products found. Try refining your query.")
