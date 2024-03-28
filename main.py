from datetime import datetime, timedelta
from utils import get_exchange_rate_history

# Get EUR to COP and USD to COP exchange rates for the past year
start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()

eur_cop = get_exchange_rate_history('EUR', 'COP', start_date, end_date)
usd_cop = get_exchange_rate_history('USD', 'COP', start_date, end_date)

# Save to CSV
eur_cop.to_csv('data/eur_cop.csv', index=False)
usd_cop.to_csv('data/usd_cop.csv', index=False)