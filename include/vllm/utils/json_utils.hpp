#pragma once
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace vllm {
namespace json {

struct Value;
using Object = std::unordered_map<std::string, Value>;
using Array  = std::vector<Value>;

struct Value {
    std::variant<std::nullptr_t, bool, int64_t, double, std::string, Array, Object> data;
    Value() : data(nullptr) {}
    Value(int64_t v) : data(v) {}
    Value(double v) : data(v) {}
    Value(const std::string& v) : data(v) {}
    Value(const char* v) : data(std::string(v)) {}
    Value(Array v) : data(std::move(v)) {}
    Value(Object v) : data(std::move(v)) {}

    bool is_object() const { return std::holds_alternative<Object>(data); }
    bool is_array() const  { return std::holds_alternative<Array>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }

    const std::string& get_string() const {
        static std::string e;
        if (auto* p = std::get_if<std::string>(&data)) return *p;
        return e;
    }
    int64_t get_int() const {
        if (auto* p = std::get_if<int64_t>(&data)) return *p;
        if (auto* p = std::get_if<double>(&data)) return static_cast<int64_t>(*p);
        return 0;
    }
    double get_number() const {
        if (auto* p = std::get_if<double>(&data)) return *p;
        if (auto* p = std::get_if<int64_t>(&data)) return static_cast<double>(*p);
        return 0.0;
    }

    const Value& operator[](const std::string& key) const {
        static Value nil;
        if (auto* obj = std::get_if<Object>(&data)) {
            auto it = obj->find(key);
            if (it != obj->end()) return it->second;
        }
        return nil;
    }
    const Value& operator[](int idx) const {
        static Value nil;
        if (auto* arr = std::get_if<Array>(&data)) {
            if (idx >= 0 && static_cast<size_t>(idx) < arr->size())
                return (*arr)[static_cast<size_t>(idx)];
        }
        return nil;
    }
    bool contains(const std::string& key) const {
        if (auto* obj = std::get_if<Object>(&data))
            return obj->find(key) != obj->end();
        return false;
    }
    size_t size() const {
        if (auto* arr = std::get_if<Array>(&data)) return arr->size();
        if (auto* obj = std::get_if<Object>(&data)) return obj->size();
        return 0;
    }
    // Range-for over objects
    const Object& get_object() const {
        static Object e;
        if (auto* o = std::get_if<Object>(&data)) return *o;
        return e;
    }
    const Array& get_array() const {
        static Array e;
        if (auto* a = std::get_if<Array>(&data)) return *a;
        return e;
    }
};

// ── Parser ─────────────────────────────────────────────────
class Parser {
    const std::string& s_;
    size_t i_ = 0;
    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    char next() { return i_ < s_.size() ? s_[i_++] : '\0'; }
    void skip_ws() { while (i_ < s_.size() && (s_[i_]==' '||s_[i_]=='\t'||s_[i_]=='\n'||s_[i_]=='\r')) i_++; }
    void expect(char c) { skip_ws(); if (next()!=c) throw std::runtime_error(std::string("Expected '")+c+"'"); }

    Value parse_val() {
        skip_ws();
        char c = peek();
        if (c=='{') return parse_obj();
        if (c=='[') return parse_arr();
        if (c=='"') return parse_str();
        if (c=='t'||c=='f') { i_+= (c=='t'?4:5); return Value(static_cast<int64_t>(c=='t'?1:0)); }
        if (c=='n') { i_+=4; return Value(); }
        return parse_num();
    }
    Value parse_obj() {
        expect('{'); Object o; skip_ws();
        if (peek()=='}') { next(); return Value(o); }
        for(;;){ skip_ws(); std::string k=parse_str().get_string(); expect(':'); o[k]=parse_val(); skip_ws(); if(peek()=='}'){next(); break;} expect(','); }
        return Value(o);
    }
    Value parse_arr() {
        expect('['); Array a; skip_ws();
        if (peek()==']') { next(); return Value(a); }
        for(;;){ a.push_back(parse_val()); skip_ws(); if(peek()==']'){next(); break;} expect(','); }
        return Value(a);
    }
    Value parse_str() {
        expect('"'); std::string r;
        while(i_<s_.size()&&s_[i_]!='"'){
            char c=s_[i_++];
            if(c=='\\'&&i_<s_.size()){ char e=s_[i_++];
                switch(e){case'"':r+='"';break;case'\\':r+='\\';break;case'/':r+='/';break;case'n':r+='\n';break;case't':r+='\t';break;case'r':r+='\r';break;
                case'u':{auto h=s_.substr(i_,4);i_+=4;int cp=std::stoi(h,nullptr,16);
                    if(cp<=0x7F)r+=static_cast<char>(cp);else if(cp<=0x7FF){r+=static_cast<char>(0xC0|(cp>>6));r+=static_cast<char>(0x80|(cp&0x3F));}
                    else{r+=static_cast<char>(0xE0|(cp>>12));r+=static_cast<char>(0x80|((cp>>6)&0x3F));r+=static_cast<char>(0x80|(cp&0x3F));}}break;
                default:r+=e;}
            }else r+=c;
        }
        expect('"'); return Value(r);
    }
    Value parse_num() {
        skip_ws(); size_t st=i_;
        if(peek()=='-')i_++;
        while(i_<s_.size()&&s_[i_]>='0'&&s_[i_]<='9')i_++;
        bool is_float=false;
        if(i_<s_.size()&&s_[i_]=='.'){ i_++; is_float=true; while(i_<s_.size()&&s_[i_]>='0'&&s_[i_]<='9')i_++; }
        if(i_<s_.size()&&(s_[i_]=='e'||s_[i_]=='E')){i_++; is_float=true; if(s_[i_]=='+'||s_[i_]=='-')i_++; while(i_<s_.size()&&s_[i_]>='0'&&s_[i_]<='9')i_++;}
        auto sub=s_.substr(st,i_-st);
        if(is_float) return Value(std::strtod(sub.c_str(),nullptr));
        return Value(static_cast<int64_t>(std::strtoll(sub.c_str(),nullptr,10)));
    }
public:
    explicit Parser(const std::string& s) : s_(s) {}
    Value parse() { return parse_val(); }
};

inline Value parse(const std::string& src) { return Parser(src).parse(); }
inline Value parse_file(const std::string& path) {
    std::ifstream f(path);
    if(!f) throw std::runtime_error("Cannot open: "+path);
    std::stringstream ss; ss<<f.rdbuf();
    return Parser(ss.str()).parse();
}

}  // namespace json
}  // namespace vllm
