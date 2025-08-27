from collections import Counter
from src.config import (
    TONE_MAP, SPECIAL_CHARACTER_REPLACE, SPECIAL_CHARACTER_BLOCK, 
    VALID_BEFORE, CLASSES, char_display_map, BASE_CHAR_MAP
)
from src.utils import special_characters_prediction

class TextProcessor:
    def __init__(self):
        self.sentence = ""
        self.current_word = ""
        self.just_processed_character = False  # Thêm thuộc tính này để tránh lỗi truy cập
        
        # Cache cho performance
        self._display_text_cache = ""
        self._full_text_cache = ""
        self._cache_dirty = True
    
    def apply_tone(self, char, tone):
        """Apply tone to a character"""
        return TONE_MAP.get(char.upper(), {}).get(tone, char)
    
    def most_common_value(self, lst):
        """Get the most common value from a list"""
        if not lst:
            return None
        return Counter(lst).most_common(1)[0][0]
    
    def process_character(self, raw_character):
        """Process and add character to current word"""
        character = special_characters_prediction(
            self.sentence + self.current_word, raw_character
        )
        
        if not character:
            return False
            
        mapped_character = char_display_map.get(character, character)
        
        if self.current_word:
            last_char = self.current_word[-1]
            
            # Kiểm tra ký tự cơ bản để giữ nguyên ký tự có dấu
            last_base_char = BASE_CHAR_MAP.get(last_char, last_char)
            current_base_char = BASE_CHAR_MAP.get(mapped_character, mapped_character)
            if last_base_char == current_base_char:
                return True  # Giữ nguyên current_word, không thêm ký tự mới
            
            # Logic xử lý ký tự đặc biệt
            if self._should_skip_character(last_char, mapped_character, raw_character):
                return False
            
            # Xử lý thay thế ký tự đặc biệt
            if self._should_replace_character(raw_character, last_char):
                self.current_word = (
                    self.current_word[:-1] + 
                    SPECIAL_CHARACTER_REPLACE[raw_character][last_char]
                )
                self._cache_dirty = True  # Đánh dấu cache cần update
                return True
            
            # Xử lý ký tự Đ
            if self._should_process_d_character(last_char, mapped_character):
                return True
            
            # Thêm ký tự mới
            self.current_word += mapped_character
            self._cache_dirty = True  # Đánh dấu cache cần update
        else:
            # Ký tự đầu tiên của từ
            if mapped_character not in ["Mu", "Munguoc", "Rau"]:
                self.current_word = mapped_character
                self._cache_dirty = True  # Đánh dấu cache cần update
        
        return True
    
    def _should_skip_character(self, last_char, mapped_character, raw_character):
        """Check if character should be skipped"""
        # Bỏ qua nếu ký tự trùng với ký tự cuối (trừ ký tự đặc biệt)
        if (last_char == mapped_character and 
            mapped_character not in ["Mu", "Munguoc", "Rau"]):
            return True
        
        # Kiểm tra ký tự đặc biệt Mu, Munguoc, Rau
        if (mapped_character == "Mu" and 
            last_char not in VALID_BEFORE["Mu"]):
            return True
        
        if (mapped_character == "Munguoc" and 
            last_char not in VALID_BEFORE["Munguoc"]):
            return True
        
        if (mapped_character == "Rau" and 
            last_char not in VALID_BEFORE["Rau"]):
            return True
        
        # Kiểm tra block characters
        if (last_char in SPECIAL_CHARACTER_BLOCK and 
            raw_character in SPECIAL_CHARACTER_BLOCK[last_char]):
            return True
        
        # Kiểm tra valid before
        if (mapped_character in VALID_BEFORE and 
            last_char not in VALID_BEFORE[mapped_character]):
            return True
        
        return False
    
    def _should_replace_character(self, raw_character, last_char):
        """Check if character should be replaced"""
        return (raw_character in SPECIAL_CHARACTER_REPLACE and 
                last_char in SPECIAL_CHARACTER_REPLACE[raw_character])
    
    def _should_process_d_character(self, last_char, mapped_character):
        """Process D and Đ characters"""
        if last_char == "D" and mapped_character == "Đ":
            self.current_word = self.current_word[:-1] + "Đ"
            return True
        
        if last_char == "Đ" and mapped_character in ["D", "DD"]:
            return True  # Skip
        
        return False
    
    def apply_tone_to_word(self, tone):
        """Apply tone to the last character of current word"""
        if self.current_word:
            last_char = self.current_word[-1]
            new_char = self.apply_tone(last_char, tone)
            self.current_word = self.current_word[:-1] + new_char
            self._cache_dirty = True  # Đánh dấu cache cần update
    
    def finalize_word(self):
        """Add current word to sentence"""
        if self.current_word:
            self.sentence += self.current_word + " "
            self.current_word = ""
            self._cache_dirty = True  # Đánh dấu cache cần update
    
    def clear_text(self):
        """Clear all text"""
        self.sentence = ""
        self.current_word = ""
        self._cache_dirty = True  # Đánh dấu cache cần update
    
    def delete_last_word(self):
        """Delete the last character from current word or sentence"""
        deleted = False
        
        if self.current_word and len(self.current_word) > 0:
            # Nếu có từ hiện tại, xóa ký tự cuối cùng
            self.current_word = self.current_word[:-1]
            self._cache_dirty = True  # Đánh dấu cache cần update
            deleted = True
        elif self.sentence and len(self.sentence) > 0:
            # Nếu có câu, xóa ký tự cuối cùng (bỏ qua dấu cách ở cuối)
            sentence = self.sentence.rstrip()  # Loại bỏ dấu cách ở cuối
            if len(sentence) > 0:
                # Xóa ký tự cuối cùng của sentence (không phải dấu cách)
                self.sentence = sentence[:-1] + " "  # Giữ lại dấu cách
            else:
                # Nếu sentence chỉ còn dấu cách, xóa hết
                self.sentence = ""
            self._cache_dirty = True  # Đánh dấu cache cần update
            deleted = True
            
        return deleted
    

    
    def get_display_text(self):
        """Get formatted display text with caching"""
        if self._cache_dirty:
            display_text = ""
            for char in self.sentence + self.current_word:
                display_text += char_display_map.get(char, char)
            self._display_text_cache = display_text.strip()
            self._cache_dirty = False
        return self._display_text_cache
    
    def get_full_text(self):
        """Get full text including current word with caching"""
        if self._cache_dirty:
            self._full_text_cache = self.sentence + self.current_word
            self._cache_dirty = False
        return self._full_text_cache