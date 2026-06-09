// =============================================================================
// NeuroFlow C++ Runtime — NPY I/O implementation
// =============================================================================

#include "neuroflow/npy_io.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nflow::npy {

namespace {

// Skip a group: "{...}" with matching braces, or "(...)" with matching parens.
std::string SkipGroup(const std::string& s, size_t& pos) {
    if (pos >= s.size()) return "";
    char open_c = s[pos];
    char close_c;
    if (open_c == '{') close_c = '}';
    else if (open_c == '(') close_c = ')';
    else return "";
    size_t start = pos;
    int depth = 0;
    while (pos < s.size()) {
        char c = s[pos];
        if (c == open_c) depth++;
        else if (c == close_c) {
            depth--;
            if (depth == 0) { pos++; return s.substr(start, pos - start); }
        }
        pos++;
    }
    return "";
}

}  // namespace

Status Read(const std::string& path, Tensor& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return Status::FileNotFound("cannot open npy: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string buf = ss.str();
    if (buf.size() < 10) return Status::ParseError("npy: too short");
    if (static_cast<unsigned char>(buf[0]) != 0x93u || buf[1] != 'N' || buf[2] != 'U' ||
        buf[3] != 'M' || buf[4] != 'P' || buf[5] != 'Y') {
        return Status::ParseError("npy: bad magic");
    }
    uint8_t major = static_cast<uint8_t>(buf[6]);
    uint8_t minor = static_cast<uint8_t>(buf[7]);
    // NPY v1 layout: magic(6) + ver_major(1) + ver_minor(1) + header_len_le(2) + header
    // NPY v2/v3: same but header_len is 4 bytes instead of 2.
    size_t header_len_off = 8;  // where the header_len integer lives
    if (major == 1) {
        // header_len is 2 bytes at offset 8
        // No endianness byte in v1 (always little-endian on x86).
    } else if (major == 2 || major == 3) {
        // header_len is 4 bytes at offset 8; followed by no endianness flag in v3
        // (v2 still has the 0x01 little-endian byte at offset 12)
        if (minor == 0 && buf[header_len_off + 4] != 0x01) {
            return Status::ParseError("npy: header not little-endian");
        }
    } else {
        return Status::ParseError("npy: unsupported version");
    }
    size_t header_len = 0;
    size_t header_len_size = (major == 1) ? 2 : 4;
    if (major == 1) {
        uint16_t hl;
        std::memcpy(&hl, buf.data() + header_len_off, 2);
        header_len = hl;
    } else {
        uint32_t hl;
        std::memcpy(&hl, buf.data() + header_len_off, 4);
        header_len = hl;
    }
    size_t data_off = header_len_off + header_len_size + header_len;
    std::string header(reinterpret_cast<const char*>(buf.data()) + data_off - header_len, header_len);
    // Parse: 'descr': '<f4', 'fortran_order': False, 'shape': (a, b, ...), ...
    auto find_key = [&](const std::string& key) -> std::string {
        std::string pat = "'" + key + "'";
        size_t pos = header.find(pat);
        if (pos == std::string::npos) return "";
        pos += pat.size();
        while (pos < header.size() && (header[pos] == ' ' || header[pos] == ':' || header[pos] == '\n'))
            pos++;
        // Find value
        if (pos < header.size() && header[pos] == '\'') {
            // string
            size_t end = header.find('\'', pos + 1);
            return header.substr(pos + 1, end - pos - 1);
        }
        if (pos < header.size() && (header[pos] == '(' || header[pos] == '{')) {
            return SkipGroup(header, pos);
        }
        if (pos < header.size() && (header[pos] == 'T' || header[pos] == 'F')) {
            return std::string(1, header[pos]);
        }
        // number
        size_t end = pos;
        while (end < header.size() && (std::isdigit(static_cast<unsigned char>(header[end])) || header[end] == '-' || header[end] == '+'))
            end++;
        return header.substr(pos, end - pos);
    };
    std::string descr = find_key("descr");
    if (descr != "<f4" && descr != "f4") {
        return Status::UnsupportedOp("npy: only float32 supported, got descr=" + descr);
    }
    std::string fortran = find_key("fortran_order");
    if (fortran == "True") {
        return Status::UnsupportedOp("npy: fortran_order not supported");
    }
    std::string shape_str = find_key("shape");
    // Parse shape tuple
    std::vector<int64_t> shape;
    size_t pos = shape_str.find('(');
    if (pos != std::string::npos) pos++;
    while (pos < shape_str.size() && shape_str[pos] != ')') {
        if (std::isdigit(static_cast<unsigned char>(shape_str[pos]))) {
            int64_t v = 0;
            while (pos < shape_str.size() && std::isdigit(static_cast<unsigned char>(shape_str[pos]))) {
                v = v * 10 + (shape_str[pos] - '0');
                pos++;
            }
            shape.push_back(v);
        } else {
            pos++;
        }
    }
    int64_t numel = 1;
    for (auto d : shape) numel *= d;
    if (buf.size() - data_off < static_cast<size_t>(numel) * sizeof(float)) {
        return Status::ParseError("npy: truncated data");
    }
    out = Tensor::Wrap(shape, reinterpret_cast<const float*>(buf.data() + data_off));
    return Status::Ok();
}

Status Write(const std::string& path, const Tensor& t) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return Status::FileNotFound("cannot write npy: " + path);
    // Header
    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < t.shape().size(); ++i) {
        if (i) hdr << ", ";
        hdr << t.shape()[i];
    }
    hdr << "), }";
    // The npy prefix (magic + version + header_len + header_text) must have a
    // total length that is a multiple of 64. prefix = 10 bytes fixed + header_len.
    std::string header = hdr.str();
    const size_t PREFIX_FIXED = 10;  // magic(6) + version(2) + header_len(2)
    size_t target = PREFIX_FIXED + header.size() + 1;  // +1 for the '\n'
    while (target % 64 != 0) {
        header += ' ';
        target++;
    }
    header += '\n';
    uint16_t header_len = static_cast<uint16_t>(header.size());
    f.put(static_cast<char>(0x93));
    f.write("NUMPY", 5);
    f.put(static_cast<char>(1));  // major
    f.put(static_cast<char>(0));  // minor
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(t.data()), t.bytes());
    return Status::Ok();
}

}  // namespace nflow::npy
