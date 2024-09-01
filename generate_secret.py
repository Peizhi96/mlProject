import secrets

secret_key = secrets.token_hex(32)

with open('.env', 'a') as f:
    f.write(f"SECRET_KEY={secret_key}\n")

print("Secret key generated and saved to .env file")