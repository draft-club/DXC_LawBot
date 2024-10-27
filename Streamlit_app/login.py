# Login functionality
'''st.header('Login')
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    response = requests.post('http://127.0.0.1:5000/login', json={'username': username, 'password': password})
    data = response.json()
    if response.status_code == 200:
        st.success('Login successful')
        if 'access_token' in data:
            st.session_state['access_token'] = data['access_token']
    else:
        st.error(data.get('error', 'Login failed'))'''