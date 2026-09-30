import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="LPBank AI RM Copilot", layout="wide")
st.title("LPBank AI Customer Analytics")


@st.cache_data
def load_data():
    file_path = "LPB_AI_Banking_Realistic_Mock_Data_v2.xlsx"
    customers = pd.read_excel(file_path, sheet_name="CUSTOMER_MASTER")
    transactions = pd.read_excel(file_path, sheet_name="TRANSACTION_DATA")
    features = pd.read_excel(file_path, sheet_name="CUSTOMER_FEATURE")
    nbo = pd.read_excel(file_path, sheet_name="NEXT_BEST_OFFER")
    signals = pd.read_excel(file_path, sheet_name="CUSTOMER_SIGNAL")
    return customers, transactions, features, nbo, signals

customers, transactions, features, nbo, signals = load_data()



customer_list = customers['CUSTOMER_ID'].tolist()
selected_cid = st.sidebar.selectbox("Chọn Mã Khách Hàng (CIF)", customer_list)

st.sidebar.divider()

api_key = st.sidebar.text_input("Nhập API Key:", type="password")


cust_info = customers[customers['CUSTOMER_ID'] == selected_cid].iloc[0]
cust_txn = transactions[transactions['CUSTOMER_ID'] == selected_cid]
cust_feat = features[features['CUSTOMER_ID'] == selected_cid].iloc[0]

#1
st.header("1. Tổng quan khách hàng")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tên Khách Hàng", cust_info['CUSTOMER_NAME'])
col2.metric("Phân Khúc", cust_info['CUSTOMER_SEGMENT'])
col3.metric("Số lượng Giao dịch (180D)", str(cust_feat['TXN_COUNT_180D']))
col4.metric("Tổng Chi Tiêu (180D)", f"{cust_feat['DEBIT_AMOUNT_180D']:,.0f} đ")

st.divider()
col_chart, col_table = st.columns([1, 1])

with col_chart:
    st.subheader("Cơ cấu chi tiêu")
    spend_by_cat = cust_txn[cust_txn['DR_CR_IND'] == 'DEBIT'].groupby('MERCHANT_CATEGORY')['AMOUNT'].sum().reset_index()
    if not spend_by_cat.empty:
        fig = px.pie(spend_by_cat, values='AMOUNT', names='MERCHANT_CATEGORY', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Khách hàng này chưa có dữ liệu phân loại chi tiêu.")

with col_table:
    st.subheader("Giao dịch gần đây")
    display_txn = cust_txn[['TRANSACTION_DATE', 'AMOUNT', 'MERCHANT_CATEGORY', 'TRANSACTION_DESCRIPTION']].head(8)
    st.dataframe(display_txn, use_container_width=True)

#2
st.divider()
st.header("2. AI Customer Insight")
cust_signals = signals[signals['CUSTOMER_ID'] == selected_cid]

if not cust_signals.empty:
    for idx, row in cust_signals.iterrows():
        st.info(f"**{row['SIGNAL_NAME']}**: {row['SIGNAL_REASON']}")
else:
    st.write("Chưa có dữ liệu phân tích hành vi cho khách hàng này.")

#3
st.header("3. Next Best Offer (Đề xuất sản phẩm)")
cust_nbo = nbo[nbo['CUSTOMER_ID'] == selected_cid]

if not cust_nbo.empty:
    for idx, row in cust_nbo.iterrows():
        with st.expander(f"Đề xuất: {row['RECOMMENDED_PRODUCT_NAME']} (Độ phù hợp: {row['NBO_SCORE']}/100)", expanded=True):
            st.write(f"- **Loại ưu đãi**: {row['OFFER_TYPE']}")
            st.write(f"- **Lý do đề xuất**: {row['NBO_REASON']}")
            st.write(f"- **Kênh tiếp cận tối ưu**: {row['ACTION_CHANNEL']}")
else:
    st.write("Không có đề xuất sản phẩm nào cho khách hàng này.")

#4
st.divider()
st.header("4. RM Copilot Chat")
st.caption("Trợ lý AI hỗ trợ phân tích chuyên sâu cho nhân viên ngân hàng.")


if "messages" not in st.session_state:
    st.session_state.messages = []


if "current_customer" not in st.session_state or st.session_state.current_customer != selected_cid:
    st.session_state.messages = []
    st.session_state.current_customer = selected_cid


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input(f"Hỏi AI về khách hàng {cust_info['CUSTOMER_NAME']}..."):
    
    # Hiển thị câu hỏi
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

 
    with st.chat_message("assistant"):
        if not api_key:
            st.error("Vui lòng nhập API !")
        else:
            try:
                import google.generativeai as genai
                
               
                genai.configure(api_key=api_key)

               
                model = genai.GenerativeModel('models/gemma-4-26b-a4b-it')

                
                signals_text = ", ".join([row['SIGNAL_NAME'] for _, row in cust_signals.iterrows()]) if not cust_signals.empty else "Không có tín hiệu đặc biệt"
                nbo_text = ", ".join([row['RECOMMENDED_PRODUCT_NAME'] for _, row in cust_nbo.iterrows()]) if not cust_nbo.empty else "Không có đề xuất sản phẩm"
                
               
                context_prompt = f"""
                Bạn là trợ lý RM Copilot của ngân hàng.
                Dữ liệu khách hàng: {cust_info['CUSTOMER_NAME']} | {cust_info['CUSTOMER_SEGMENT']} | {cust_feat['DEBIT_AMOUNT_180D']} VND | Insight: {signals_text} | NBO: {nbo_text}
                
                Câu hỏi: {prompt}
                
                YÊU CẦU TỐI THƯỢNG: Bắt đầu câu trả lời bằng đúng cụm từ "[TRẢ LỜI]:" (viết hoa, có ngoặc vuông và dấu hai chấm). Sau đó đưa ra luôn lời tư vấn tiếng Việt ngắn gọn. Tuyệt đối không viết thêm bất cứ chữ tiếng Anh nào phía trên.
                """

            
                gemini_history = []
                for msg in st.session_state.messages[:-1]:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append({"role": role, "parts": [msg["content"]]})

                
                chat = model.start_chat(history=gemini_history)
                response = chat.send_message(context_prompt)
                
                raw_answer = response.text
                
               
                if "[TRẢ LỜI]:" in raw_answer:
                    
                    answer = raw_answer.split("[TRẢ LỜI]:")[-1].strip()
                elif "<FINAL_ANSWER>" in raw_answer:
                    
                    answer = raw_answer.split("<FINAL_ANSWER>")[-1].split("</FINAL_ANSWER>")[0].strip()
                else:
                    
                    import re
                    clean_text = re.sub(r'```.*?```', '', raw_answer, flags=re.DOTALL)
                    lines = [line.strip() for line in clean_text.split('\n') if line.strip() and len(line.strip()) > 20]
                    answer = lines[-1] if lines else raw_answer

                
                answer = answer.replace('"', '').replace('*', '').replace('`', '').strip()

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi kết nối API: {e}")