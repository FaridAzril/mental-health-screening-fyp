#!/usr/bin/env python3
"""
Test Supabase Connection and Authentication
"""
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Your Supabase credentials
# Replace with your actual credentials
url = "https://pgyimgczgrcplyoiimhl.supabase.co"  # Replace with your project URL
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBneWltZ2N6Z3JjcGx5b2lpbWhsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY3NTYzNDcsImV4cCI6MjA5MjMzMjM0N30.Yu27XajkuXSGpSdj326Fx-0cShtArbp9bNH2XQta3DI"  # Replace with your anon key

# Create Supabase client
try:
    supabase: Client = create_client(url, key)
    print("✅ Supabase client created successfully!")
except Exception as e:
    print(f"❌ Error creating Supabase client: {e}")
    exit(1)

# Test authentication with admin user
def test_auth():
    """Test admin authentication"""
    try:
        # Sign in with admin credentials
        auth_response = supabase.auth.sign_in_with_password({
            'email': 'admin@hospital.com',
            'password': 'Farid01.'  # Use your actual password
        })
        
        if auth_response.user:
            print(f"✅ Authentication successful!")
            print(f"   User ID: {auth_response.user.id}")
            print(f"   Email: {auth_response.user.email}")
            
            # Test database access
            test_database_access(auth_response.user.id)
            return True
        else:
            print("❌ Authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False

def test_database_access(user_id):
    """Test database access and role verification"""
    try:
        # Debug: Show all profiles in table
        all_profiles = supabase.table('user_profiles').select('*').execute()
        print(f"🔍 Debug: All user_profiles in table:")
        for profile in all_profiles.data:
            print(f"   ID: {profile.get('id')}, Email: {profile.get('email')}, Role: {profile.get('role')}")
        
        # Test user profile access
        profile_response = supabase.table('user_profiles').select('*').eq('id', user_id).execute()
        
        if profile_response.data:
            profile = profile_response.data[0]
            print(f"✅ Database access successful!")
            print(f"   Role: {profile.get('role')}")
            print(f"   Name: {profile.get('name')}")
            print(f"   Email: {profile.get('email')}")
            
            # Test if role is properly set
            if profile.get('role') == 'admin':
                print("✅ Admin role confirmed!")
            else:
                print(f"⚠️  Role found: {profile.get('role')}")
        else:
            print("❌ No user profile found")
            
    except Exception as e:
        print(f"❌ Database access error: {e}")

def test_table_access():
    """Test access to all tables"""
    try:
        # Test doctors table
        doctors = supabase.table('doctors').select('count').execute()
        print(f"✅ Doctors table accessible: {len(doctors.data)} records")
        
        # Test patients table
        patients = supabase.table('patients').select('count').execute()
        print(f"✅ Patients table accessible: {len(patients.data)} records")
        
        # Test screenings table
        screenings = supabase.table('screenings').select('count').execute()
        print(f"✅ Screenings table accessible: {len(screenings.data)} records")
        
    except Exception as e:
        print(f"❌ Table access error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Supabase Connection and Authentication")
    print("=" * 50)
    
    # Test authentication
    if test_auth():
        print("\n🗄️ Testing Database Access")
        print("=" * 50)
        test_table_access()
        
        print("\n🎯 All tests completed!")
        print("✅ Supabase is ready for Flask integration!")
    else:
        print("\n❌ Authentication failed - check credentials")
