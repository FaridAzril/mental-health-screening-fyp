"""
Working Supabase Client Configuration
"""
import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://pgyimgczgrcplyoiimhl.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBneWltZ2N6Z3JjcGx5b2lpbWhsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY3NTYzNDcsImV4cCI6MjA5MjMzMjM0N30.Yu27XajkuXSGpSdj326Fx-0cShtArbp9bNH2XQta3DI')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_SERVICE_KEY', None)
# Try to read from .env.example if .env doesn't exist
if not SUPABASE_SERVICE_KEY:
    try:
        with open('.env.example', 'r') as f:
            for line in f:
                if line.startswith('SUPABASE_SERVICE_KEY='):
                    SUPABASE_SERVICE_KEY = line.split('=', 1)[1].strip()
                    break
    except:
        pass

# Create Supabase client using HTTP directly
class WorkingSupabaseClient:
    def __init__(self, url, key):
        self.url = url
        self.key = key
        self.headers = {
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'
        }
        self.client = httpx.Client()
    
    @property
    def auth(self):
        """Return auth sub-client"""
        parent = self
        class AuthClient:
            def sign_in_with_password(self, credentials):
                response = parent.client.post(
                    f"{parent.url}/auth/v1/token?grant_type=password",
                    json=credentials,
                    headers=parent.headers
                )
                return response
            
            def sign_out(self):
                pass
        return AuthClient()
    
    def table(self, table_name):
        class TableClient:
            def __init__(self, client, table_name):
                self.client = client
                self.table_name = table_name
            
            def select(self, columns='*'):
                class SelectClient:
                    def __init__(self, client, table_name, columns):
                        self.client = client
                        self.table_name = table_name
                        self.columns = columns
                    
                    def execute(self, auth_token=None):
                        headers = dict(self.client.headers)
                        if auth_token:
                            headers['Authorization'] = f'Bearer {auth_token}'
                        response = self.client.client.get(
                            f"{self.client.url}/rest/v1/{self.table_name}?select={self.columns}",
                            headers=headers
                        )
                        return response
                    
                    def eq(self, field, value):
                        self.columns = f"{self.columns}&{field}=eq.{value}"
                        return self
                    
                    def order(self, field, desc=False):
                        direction = 'desc' if desc else 'asc'
                        self.columns = f"{self.columns}&order={field}.{direction}"
                        return self
                
                return SelectClient(self.client, table_name, columns)
            
            def insert(self, data):
                class InsertClient:
                    def __init__(self, client, table_name, data):
                        self.client = client
                        self.table_name = table_name
                        self.data = data
                    
                    def execute(self, auth_token=None):
                        headers = dict(self.client.headers)
                        if auth_token:
                            headers['Authorization'] = f'Bearer {auth_token}'
                        response = self.client.client.post(
                            f"{self.client.url}/rest/v1/{self.table_name}",
                            json=self.data,
                            headers=headers
                        )
                        return response
                
                return InsertClient(self.client, table_name, data)
        
        return TableClient(self, table_name)

# Create the working client
supabase_client = WorkingSupabaseClient(SUPABASE_URL, SUPABASE_KEY)
print("Working Supabase client initialized successfully!")

# Export for use in other modules
supabase = supabase_client
