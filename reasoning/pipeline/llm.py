"""Minimal OpenAI-compatible client. No SDK, no credentials in this repository.

The endpoint is supplied by the operator through the same environment variables the demo
runner uses, so one configured model serves both. Model output is parsed as JSON and
never executed, imported or eval'd; a code fence is tolerated because servers add one, and
nothing else is.
"""
import json
import os
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlsplit

MAX_RESPONSE = 4 * 1024 * 1024


class LLMError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise LLMError('Model endpoint attempted a redirect. Configure the final URL.')


class Client:
    def __init__(self, base_url=None, model=None, api_key=None, timeout=90, temperature=0,
                 max_tokens=2500):
        self.base_url = (base_url or os.environ.get('UW_LLM_BASE_URL', '')).rstrip('/')
        self.model = model or os.environ.get('UW_LLM_MODEL', '')
        self.api_key = api_key if api_key is not None else os.environ.get('UW_LLM_API_KEY', '')
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def configured(self):
        return bool(self.base_url and self.model)

    def describe(self):
        return {'configured': self.configured, 'base_url_set': bool(self.base_url),
                'model': self.model or None,
                'note': 'Set UW_LLM_BASE_URL and UW_LLM_MODEL. Reachability is tested by a real call.'}

    def complete(self, messages):
        if not self.configured:
            raise LLMError('Set UW_LLM_BASE_URL and UW_LLM_MODEL before running the reasoning step.')
        url = self.base_url + '/chat/completions'
        if urlsplit(url).scheme not in ('http', 'https'):
            raise LLMError('Model endpoint must use HTTP or HTTPS.')
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        payload = {'model': self.model, 'messages': messages, 'temperature': self.temperature,
                   'max_tokens': self.max_tokens}
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE + 1)
        except urllib.error.HTTPError as error:
            raise LLMError(f'Model endpoint returned HTTP {error.code}.') from None
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            raise LLMError('Model endpoint could not be reached or timed out.') from None
        if len(raw) > MAX_RESPONSE:
            raise LLMError('Model response exceeded 4 MB.')
        try:
            body = json.loads(raw)
            content = body['choices'][0]['message']['content']
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError('Model response lacked chat completion content.') from None
        if not isinstance(content, str):
            raise LLMError('Model returned non-text content.')
        return parse_json(content)


def parse_json(content):
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip())
    try:
        parsed = json.loads(cleaned)
    except ValueError:
        raise LLMError('Model output was not valid JSON.') from None
    if not isinstance(parsed, dict):
        raise LLMError('Model must return a JSON object.')
    return parsed
