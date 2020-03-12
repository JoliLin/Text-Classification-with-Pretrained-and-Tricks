import unicodedata

def _is_punctuation(char):
	cp = ord(char)
	if((cp>=33 and cp<=47) or (cp>=58 and cp<=64) or(cp>=91 and cp<=96) or (cp>=123 and cp<=126)):
		return True

	cat = unicodedata.category(char)
	if cat.startswith('P'):
		return True
	return False

def _run_strip_accents(text):
	text = unicodedata.normalize('NFD', text)
	output = []
	for char in text:
		cat = unicodedata.category(char)
		if cat =='Mn':
			continue
		output.append(char)
	return ''.join(output)

def _run_split_on_punc(text):
	chars = list(text)
	i = 0
	start_new_word = True
	output = []
	while i < len(chars):
		char = chars[i]
		if _is_punctuation(char):
			output.append([char])
			start_new_word =True
		else:
			if start_new_word:
				output.append([])
			start_new_word = False
			output[-1].append(char)
		i += 1
	return [''.join(x) for x in output]

def whitespace_tokenize(text):
	text = text.strip()
	if not text:
		return []
	tokens = text.split()
	return tokens

def convert2unicode(text):
	if isinstance(text, str):
		return str
	elif isinstance(text, bytes):
		return text.decode('utf-8', 'ignore')
	else:
		raise ValueError('Unsupported string type: %s' % (type(text)))

def tokenize(text):
	org_tokens = whitespace_tokenize(text)
	split_tokens = []
	for token in org_tokens:
		token = token.lower()
		token = _run_strip_accents(token)
		split_tokens.extend(_run_split_on_punc(token))

	return whitespace_tokenize(' '.join(split_tokens))

def remove_stopwords(text):
	#text: list
	output_tokens = []

	for i in text:
		_ = list(map(_is_punctuation, i))
		word = []
		for w, p in zip(i, _):
			if p == False:
				word.append(w)
		
		output_tokens.append(''.join(word))

	return list(filter(None, output_tokens))
