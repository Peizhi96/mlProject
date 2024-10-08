from __future__ import print_function

from flask import Flask, request, render_template, redirect, url_for, flash, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
from itsdangerous import URLSafeTimedSerializer
from flask_migrate import Migrate


import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

load_dotenv()
 
application=Flask(__name__)

app = application

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('SQLALCHEMY_DATABASE_URI')# change it later
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# define the User model
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"
    
class UserToken(db.Model):
    __tablename__ = 'user_tokens'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    access_token = db.Column(db.String(500))
    refresh_token = db.Column(db.String(500))
    token_uri = db.Column(db.String(255))
    scopes = db.Column(db.String(500))
    expiry = db.Column(db.DateTime)

@login_manager.user_loader
def load_user(user_id):
    with current_app.app_context():
        return db.session.get(User, int(user_id))
    
def save_token_to_db(user, creds):
    token_record = UserToken.query.filter_by(user_id=user.id).first()

    if not token_record:
        token_record = UserToken(
            user_id=user.id,
            access_token=creds.token,
            refresh_token=creds.refresh_token,
            token_uri=creds.token_uri,
            scopes=",".join(creds.scopes),
            expiry=creds.expiry
        )
        db.session.add(token_record)
    else:
        token_record.access_token = creds.token
        token_record.refresh_token = creds.refresh_token
        token_record.token_uri = creds.token_uri
        token_record.scopes = ",".join(creds.scopes)
        token_record.expiry = creds.expiry
    
    db.session.commit()

# route for a register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # check if the user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!', category='error')
            return redirect(url_for('register'))

        # create a new user and save it to the database
        user = User(username=username, email=email, password=bcrypt.generate_password_hash(password).decode('utf-8'))
        db.session.add(user)
        db.session.commit()

        flash('Account created successfully!', category='success')
        return redirect(url_for('login'))

    return render_template('register.html')

# route for a login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # check if the user exists
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', category='success')
            return redirect(url_for('predict_data'))
        else:
            flash('Login failed! Please check your credentials.', category='error')
            return redirect(url_for('login'))

    return render_template('login.html')

# route for a logout page
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out!', 'success')
    return redirect(url_for('login'))

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_request():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()  
        if user:
            send_email_via_gmail(user)  
            flash('An email has been sent with instructions to reset your password.', 'info')
            return redirect(url_for('login'))
        else:
            flash('This email is not associated with any account.', 'warning')

    return render_template('reset_request.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)  
    except:
        flash('The token is either invalid or has expired.', 'warning')
        return redirect(url_for('reset_request'))

    user = User.query.filter_by(email=email).first()
    if request.method == 'POST':
        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_token.html')

def send_email_via_gmail(user):

    token_record = UserToken.query.filter_by(user_id=user.id).first()

    if token_record:
        creds = Credentials(
            token=token_record.access_token,
            refresh_token=token_record.refresh_token,
            token_uri=token_record.token_uri,
            client_id='your-client-id',
            client_secret='your-client-secret',
            scopes=token_record.scopes.split(',')
        )
    else:
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)  
            creds = flow.run_local_server(port=5002)
            save_token_to_db(user, creds)
    try:
        service = build('gmail', 'v1', credentials=creds)
        token = s.dumps(user.email, salt='password-reset-salt')
        reset_url = url_for('reset_token', token=token, _external=True)
        body = f'''Dear {user.username},

To reset your password, please click the link below:
{reset_url}

If you did not request a password reset, please ignore this email.

Best regards,
Your App Team
'''
        message = MIMEText(body)  
        message['to'] = user.email
        message['from'] = 'yanzhenyi123@gmail.com'  
        message['subject'] = 'Test Email via Gmail API'  
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        sent_message = (service.users().messages().send(userId="me", body={'raw': raw}).execute())
        print(f'Email sent successfully! Message ID: {sent_message["id"]}')
    except Exception as error:
        print(f'An error occurred: {error}')

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['Get', 'POST'])
@login_required
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            test_preparation_course=request.form.get('test_preparation_course'),
            lunch=request.form.get('lunch'),
            writing_score=int(request.form.get('writing_score')),
            reading_score=int(request.form.get('reading_score'))
        )
        
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        
        results = predict_pipeline.predict(pred_df)
        if results > 100:
            results = ["The score is greater than 100. Please check the input values"]
        print(results)
        return render_template('home.html', results=results[0])
    


if __name__ == '__main__':
    app.run(port=5001, debug=True)