import requests

def get_phone_number_info(phone_number, api_key):
    url = f"http://apilayer.net/api/validate?access_key={api_key}&number={phone_number}"
    response = requests.get(url)
    
    # Print the full response for debugging purposes
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success', True):
            return data
        else:
            return {'error': data.get('error', {}).get('info', 'Unknown error')}
    else:
        return {'error': 'Failed to reach the API'}

if __name__ == "__main__":
    api_key = 'your_numverify_api_key'  # Replace with your actual API key
    phone_number = input("Enter a phone number with country code (e.g., +14155552671): ")
    info = get_phone_number_info(phone_number, api_key)
    
    if "error" in info:
        print(f"Error: {info['error']}")
    else:
        print(f"Number: {info['number']}")
        print(f"Country: {info['country_name']}")
        print(f"Location: {info['location']}")
        print(f"Carrier: {info['carrier']}")
        print(f"Line Type: {info['line_type']}")
