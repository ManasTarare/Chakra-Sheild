import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

# --- CONFIG ---
SEED = 42
NUM_USERS = 450
NUM_RECORDS = 13500
ANOMALY_PERCENT = 0.01
DEPARTMENTS = ["IT", "HR", "Finance", "Operations", "Sales"]
FAKE_APPS_COUNT = 15
REMOTE_ACCESS_TOOL_COUNT = 2
NETWORK_IP_COUNT = 5
NETWORK_DOMAIN_COUNT = 5
MALICIOUS_SITE = "malicious-site.net"

# Date range for dataset generation (converted to datetime objects)
START_DATE = datetime(2025, 8, 1)
END_DATE = datetime(2025, 8, 30)

# --- INIT ---
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# --- USERS & DEPARTMENTS ---
users = [f"U{str(i+1).zfill(3)}" for i in range(NUM_USERS)]
user_departments = {u: DEPARTMENTS[i % len(DEPARTMENTS)] for i, u in enumerate(users)}

# --- ANOMALOUS USERS ---
anomalous_users = set(random.sample(users, max(int(NUM_USERS * ANOMALY_PERCENT), 1)))

# --- APPS & NETWORK ---
fake_apps = [fake.file_name(extension='exe') for _ in range(FAKE_APPS_COUNT)]
remote_access_tools = random.sample(fake_apps, k=REMOTE_ACCESS_TOOL_COUNT)
fake_network_sites = (
    [fake.ipv4() for _ in range(NETWORK_IP_COUNT)] +
    [fake.domain_name() for _ in range(NETWORK_DOMAIN_COUNT)] +
    [MALICIOUS_SITE]
)

def generate_login_logout(is_anomalous):
    """Generate realistic login/logout times, with anomalies for night-time activity."""
    if is_anomalous and random.random() < 0.7:
        login_hour = random.choice([0, 1, 2, 3, 22, 23])
        login_time = fake.date_time_between(start_date=START_DATE, end_date=END_DATE).replace(
            hour=login_hour, minute=random.randint(0, 59), second=0, microsecond=0
        )
        logout_time = login_time + timedelta(minutes=random.randint(30, 300))
    else:
        login_time = fake.date_time_between(start_date=START_DATE, end_date=END_DATE).replace(
            hour=9, minute=random.randint(0, 59), second=0, microsecond=0
        )
        logout_time = login_time.replace(hour=17, minute=random.randint(0, 59))
    return login_time, logout_time

def inject_anomalies(record):
    """Modify a normal record to simulate an insider threat scenario."""
    record.update({
        "failed_login_attempts": random.randint(4, 7),
        "usb_inserted": 1,
        "files_copied_to_usb": random.randint(10, 40),
        "bluetooth_usage": 1,
        "clipboard_usage": record["clipboard_usage"] + random.randint(10, 20),
        "command_shell_usage": record["command_shell_usage"] + random.randint(5, 15),
        "files_deleted": record["files_deleted"] + random.randint(5, 10),
        "data_uploaded": round(record["data_uploaded"] + random.uniform(500, 1000), 2),
        "data_downloaded": round(record["data_downloaded"] + random.uniform(1000, 3000), 2),
        "remote_access_tool": 1
    })
    sites = record["network_sites"].split(",")
    if MALICIOUS_SITE not in sites:
        sites.append(MALICIOUS_SITE)
    record["network_sites"] = ",".join(sites)
    return record

def generate_record(user):
    """Generate one synthetic endpoint activity record."""
    dept = user_departments[user]
    is_anomalous = user in anomalous_users
    login_time, logout_time = generate_login_logout(is_anomalous)

    record = {
        "user_id": user,
        "department": dept,
        "login_time": login_time,
        "logout_time": logout_time,
        "failed_login_attempts": random.choices([0, 1, 2, 3], weights=[0.7, 0.2, 0.07, 0.03])[0],
        "usb_inserted": random.choices([0, 1], weights=[0.9, 0.1])[0],
        "files_copied_to_usb": 0,
        "bluetooth_usage": random.choices([0, 1], weights=[0.9, 0.1])[0],
        "clipboard_usage": random.randint(0, 10),
        "print_usage": random.randint(0, 10),
        "command_shell_usage": random.choices([0, 1, 2, 5], weights=[0.6, 0.25, 0.1, 0.05])[0],
        "files_accessed": random.randint(1, 30),
        "files_deleted": 0,
        "data_uploaded": round(random.uniform(10, 500), 2),
        "data_downloaded": round(random.uniform(10, 1000), 2),
    }

    if record["usb_inserted"]:
        record["files_copied_to_usb"] = random.randint(0, 20)

    app_usage = random.sample(fake_apps, k=random.randint(2, 5))
    record["application_usage"] = ",".join(app_usage)
    record["remote_access_tool"] = int(any(app in remote_access_tools for app in app_usage))

    network_sample = random.sample(fake_network_sites, k=random.randint(1, 4))
    record["network_sites"] = ",".join(network_sample)

    if is_anomalous and random.random() < 0.8:
        record = inject_anomalies(record)

    return record

# --- GENERATE DATA ---
data = [generate_record(random.choice(users)) for _ in range(NUM_RECORDS)]
df = pd.DataFrame(data)
df[["login_time", "logout_time"]] = df[["login_time", "logout_time"]].apply(pd.to_datetime)
df.to_csv("activity_data.csv", index=False)
print(f"✅ Dataset saved as 'activity_data.csv' ({len(df)} records) from {START_DATE.date()} to {END_DATE.date()}")
