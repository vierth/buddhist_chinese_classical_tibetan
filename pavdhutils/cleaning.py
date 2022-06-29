"""Functions for cleaning unwanted characters from text"""

import re, os

toremove = ['』','。', '！', '，', '：', '、', '（', '）', '；', '？', '〉', '〈', '」', '「', '『', '“', '”', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '_', '`''{', '|', '}', '~', '¤', '±', '·', '×', 'à', 'á', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', '÷', 'ù', 'ú', 'ü', 'ā', 'ī', 'ń', 'ň', 'ō', 'ū', 'ǎ', 'ǐ', 'ǔ', 'ǖ', 'ǘ', 'ǚ', 'ǜ', 'ǹ', 'ɑ', 'ɡ', 'α', 'β', 'γ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'а', 'б', 'в', 'г', 'д', 'е', 'к', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', '—', '‖', '‘', '’', '…', '※', 'ⅰ', 'ⅲ', '∈', '∏', '∑', '√', '∠', '∥', '∧', '∩', '∪', '∫', '∮', '∶', '∷', '∽', '≈', '≌', '≡', '⊙', '⊥', '⌒', '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', '⑴', '⑵', '⑶', '⑷', '⑸', '⑹', '⑺', '⑻', '⑼', '⑽', '⑾', '⑿', '⒀', '⒁', '⒂', '⒃', '⒄', '⒅', '⒆', '⒈', '⒉', '⒊', '⒋', '⒌', '⒍', '⒎', '⒏', '⒐', '⒑', '⒒', '⒓', '⒔', '⒕', '⒖', '⒗', '⒘', '⒙', '⒚', '⒛', '─', '┅', '┋', '┌', '┍', '┎', '┏', '┐', '┑', '┒', '┓', '└', '┕', '┘', '┙', '┚', '┛', '├', '┝', '┞', '┠', '┡', '┢', '┣', '┤', '┥', '┦', '┧', '┩', '┪', '┫', '┬', '┭', '┮', '┯', '┰', '┱', '┲', '┳', '■', '□', '▲', '△', '◆', '◇', '○', '◎', '●', '★','︶', '﹑', '﹔', '﹖', '＂', '＃', '％', '＆', '＊','．', '／', '０', '１', '２', '３', '４', '５', '６', '７', '８', '９', '＜', '＝', '＞', '＠', '［', '＼', '］', '＿', '｀', 'ａ', 'ｂ', 'ｃ', 'ｄ', 'ｅ', 'ｆ', 'ｇ', 'ｈ', 'ｉ', 'ｊ', 'ｋ', 'ｌ', 'ｍ', 'ｎ', 'ｏ', 'ｐ', 'ｑ', 'ｒ', 'ｓ', 'ｔ', 'ｕ', 'ｖ', 'ｗ', 'ｘ', 'ｙ', 'ｚ', '｛', '｝', '～', '￥','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','《', '》', '〔', '〕', '【', '】', 'A',  'B',  'C',  'D',  'E',  'F',  'G',  'H', 'I', 'J', 'K', 'L', "M", 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',  'Ｗ',  'Ｘ',  'Ｙ',  'Ｚ',  '＾',  '｜', '￠',  '￡', '~','¶', '　']

punc_marks = "，。、？！「」《》"

# clean the text. This will remove everything in the above list from the text
def clean(content,remove=toremove,delwhitespace=True,delnewline=True, 
            extrarem="~BREAK~",otherregex=None, preserve_punc=False, 
            preserve_double_graph=False, delnonnewline=False):
    # These two lines are useful for Chinese texts where there was no whitespace or punctuation
    # in the original documents
    
    if delwhitespace:
        content = re.sub(r'\s+', '', content)
    if delnewline:
        if preserve_double_graph:
            content = re.sub(r'\n\n', '***doublegraph***', content)
            content = re.sub(r'\n', '', content)
            content = re.sub(r'\*\*\*doublegraph\*\*\*', '\n', content)
        else:
            content = re.sub(r'\n', '', content)
    if delnonnewline:
        content = re.sub(r'[ \t]', '', content)

    if extrarem in content:
        content = content.remove(extrarem, '')

    addedregex=[r'<.+?>', r'¶']
    for regex in addedregex:
        content = re.sub(regex,'',content)


    if otherregex:
        for regex in otherregex:
            content = re.sub(regex,'',content)

    if preserve_punc:
        remove = [r for r in remove if r not in punc_marks]

    for item in remove:
        content = content.replace(item, "")

    return content