try:
	from .fastchat_client import FastChatAgent
except Exception:
	# Optional dependency; FastChatAgent will be unavailable if fastchat isn't installed
	FastChatAgent = None

from .http_agent import HTTPAgent
