#![allow(non_camel_case_types)]

use std::fs::OpenOptions;

use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;

use crate::constants::{GGML_MAX_DIMS, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};

#[derive(Debug)]
pub enum gguf_type {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
    // GGUF_TYPE_COUNT, // marks the end of the enum
}

#[derive(Debug)]
// NOTE: always add types at the end of the enum to keep backward compatibility
pub enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    // GGML_TYPE_COUNT,
}

#[derive(Debug)]
pub enum gguf_value {
    uint32(u32),
    int32(i32),
    uint64(u64),
    int64(i64),
    float32(f32),
    string(String),
    array(Vec<gguf_value>),
}

#[derive(Debug)]
pub struct gguf_kv {
    pub key: String,
    pub kv_type: gguf_type,
    pub value: gguf_value,
}

#[derive(Debug)]
pub struct gguf_header {
    pub magic: [u8; 4],
    pub version: u32,
    pub n_tensors: u64, // GGUFv2
    pub n_kv: u64,      // GGUFv2
}

#[derive(Debug)]
pub struct gguf_tensor_info {
    pub name: String,
    pub n_dims: u32,
    pub ne: [u64; GGML_MAX_DIMS],
    pub t_type: ggml_type,
    pub offset: u64, // offset from start of `data`, must be a multiple of `ALIGNMENT`

                     // for writing API
                     // const void * data;
                     // size: usize,
}

#[derive(Debug)]
pub struct gguf_context {
    pub mapped_file: Mmap,

    pub header: gguf_header,
    pub kv: Vec<gguf_kv>,
    pub infos: Vec<gguf_tensor_info>,
    pub alignment: u32,
    pub offset: usize, // offset of `data` from beginning of file
    pub size: usize,   // size of `data` in bytes

                       // //uint8_t * padding;
                       // void * data;
}

pub fn gguf_find_tensor(ctx: &gguf_context, name: &str) -> usize {
    let n_tensor_info = ctx.infos.len();
    for i in 0..n_tensor_info {
        if ctx.infos[i].name == name {
            return i;
        }
    }
    panic!("invalid tensor {:?}", name);
}

pub fn load_gguf_file(model_file: &str) -> gguf_context {
    let file = OpenOptions::new()
        .read(true)
        .open(model_file)
        .expect(&format!("Couldn't open file {}\n", model_file));
    let mapped_file = unsafe { Mmap::map(&file).unwrap() };

    let mut offset = 0;
    // read the header
    let mut magic: [u8; 4] = [0; 4];
    magic.copy_from_slice(&mapped_file[offset..offset + 4]);
    for i in 0..GGUF_MAGIC.len() {
        if magic[i] != GGUF_MAGIC[i] {
            panic!("invalid magic characters {:?}", magic);
        }
    }
    offset += 4;
    let version = gguf_read_u32(&mapped_file, &mut offset);
    let n_tensors = gguf_read_u64(&mapped_file, &mut offset);
    let n_kv = gguf_read_u64(&mapped_file, &mut offset);
    let gguf_header = gguf_header {
        magic,
        version,
        n_tensors,
        n_kv,
    };
    println!("gguf_header: {:?}", gguf_header);

    // read the kv pairs
    let mut kv_vec: Vec<gguf_kv> = Vec::with_capacity(gguf_header.n_kv as usize);
    for _i in 0..gguf_header.n_kv {
        let key = gguf_read_str(&mapped_file, &mut offset);
        let kv_type = get_gguf_kv_type(gguf_read_u32(&mapped_file, &mut offset));

        let value = match kv_type {
            gguf_type::GGUF_TYPE_STRING => {
                gguf_value::string(gguf_read_str(&mapped_file, &mut offset))
            }
            gguf_type::GGUF_TYPE_UINT32 => {
                gguf_value::uint32(gguf_read_u32(&mapped_file, &mut offset))
            }
            gguf_type::GGUF_TYPE_INT32 => {
                gguf_value::int32(gguf_read_i32(&mapped_file, &mut offset))
            }
            gguf_type::GGUF_TYPE_UINT64 => {
                gguf_value::uint64(gguf_read_u64(&mapped_file, &mut offset))
            }
            gguf_type::GGUF_TYPE_INT64 => {
                gguf_value::int64(gguf_read_i64(&mapped_file, &mut offset))
            }
            gguf_type::GGUF_TYPE_FLOAT32 => {
                gguf_value::float32(gguf_read_f32(&mapped_file, &mut offset))
            }
            gguf_type::GGUF_TYPE_ARRAY => {
                let arr_type = get_gguf_kv_type(gguf_read_u32(&mapped_file, &mut offset));
                let arr_n = gguf_read_u64(&mapped_file, &mut offset);

                let mut arr_vec: Vec<gguf_value> = Vec::with_capacity(arr_n as usize);
                match arr_type {
                    gguf_type::GGUF_TYPE_STRING => {
                        for _j in 0..arr_n {
                            arr_vec
                                .push(gguf_value::string(gguf_read_str(&mapped_file, &mut offset)))
                        }
                    }
                    gguf_type::GGUF_TYPE_INT32 => {
                        for _j in 0..arr_n {
                            arr_vec
                                .push(gguf_value::int32(gguf_read_i32(&mapped_file, &mut offset)))
                        }
                    }
                    _ => panic!("unsupport arr type name={:?} - {:?}", key, arr_type),
                };
                gguf_value::array(arr_vec)
            }
            _ => panic!("unsupport kv type name={:?} - {:?}", key, kv_type),
        };
        match &value {
            gguf_value::array(arr_vec) => {
                println!(
                    "gguf_kv: {:?} {:?} => {:?}",
                    key,
                    arr_vec.len(),
                    &arr_vec[0..10]
                )
            }
            _ => println!("gguf_kv: {:?} = {:?}", key, value),
        };

        kv_vec.push(gguf_kv {
            key,
            kv_type,
            value,
        })
    }

    // read the tensor infos
    let mut tensor_info_vec: Vec<gguf_tensor_info> =
        Vec::with_capacity(gguf_header.n_tensors as usize);
    for _i in 0..gguf_header.n_tensors {
        let ti_name = gguf_read_str(&mapped_file, &mut offset);
        let ti_n_dims = gguf_read_u32(&mapped_file, &mut offset);

        let mut ne: [u64; GGML_MAX_DIMS] = [1; GGML_MAX_DIMS];
        for j in 0..ti_n_dims as usize {
            ne[j] = gguf_read_u64(&mapped_file, &mut offset);
        }
        let ti_type = get_gguf_ggml_type(gguf_read_u32(&mapped_file, &mut offset));
        let ti_offset = gguf_read_u64(&mapped_file, &mut offset);

        println!(
            "tensor_info: {:?} => {:?} {:?} {:?} {:?} {:?}",
            ti_name, ti_n_dims, ne, ti_type, ti_offset, offset
        );

        tensor_info_vec.push(gguf_tensor_info {
            name: ti_name,
            n_dims: ti_n_dims,
            ne,
            t_type: ti_type,
            offset: ti_offset,
        })
    }

    let mut ctx = gguf_context {
        mapped_file: mapped_file,
        header: gguf_header,
        kv: kv_vec,
        infos: tensor_info_vec,
        alignment: 0,
        offset: 0,
        size: 0,
    };

    ctx.alignment = GGUF_DEFAULT_ALIGNMENT;
    if let Some(_) = gguf_find_key(&ctx, "general.alignment") {
        ctx.alignment = gguf_get_val_u32(&ctx, "general.alignment");
    }

    // we require the data section to be aligned, so take into account any padding
    let offset_pad = offset as u32 % ctx.alignment;

    if offset_pad != 0 {
        offset += (ctx.alignment - offset_pad) as usize;
    }
    // store the current file offset - this is where the data section starts
    ctx.offset = offset;

    // compute the total size of the data section, taking into account the alignment
    ctx.size = 0;
    ctx.infos.iter().for_each(|tensor_info| {
        let ne = tensor_info.ne[0] * tensor_info.ne[1] * tensor_info.ne[2] * tensor_info.ne[3];
        let size_cur = ggml_row_size(&tensor_info.t_type, ne);
        ctx.size += ggml_pad(size_cur, ctx.alignment);
    });
    println!(
        "gguf_ctx.alignment={:?}, gguf_ctx.offset={:?}, gguf_ctx.size={:?}",
        ctx.alignment, ctx.offset, ctx.size
    );

    return ctx;
}

pub fn ggml_pad(size_cur: usize, alignment: u32) -> usize {
    (size_cur + alignment as usize - 1) & (!0 ^ (alignment as usize - 1))
}

pub fn ggml_row_size(g_type: &ggml_type, ne: u64) -> usize {
    return ggml_type_size(g_type) * ne as usize / ggml_blck_size(g_type);
}

pub fn ggml_type_size(g_type: &ggml_type) -> usize {
    match g_type {
        ggml_type::GGML_TYPE_F32 => 4,
        ggml_type::GGML_TYPE_F16 => 2,
        _ => 3,
    }
}

pub fn ggml_blck_size(g_type: &ggml_type) -> usize {
    match g_type {
        ggml_type::GGML_TYPE_F32 => 1,
        ggml_type::GGML_TYPE_F16 => 1,
        _ => 3,
    }
}

pub fn get_gguf_ggml_type(ggml_type: u32) -> ggml_type {
    match ggml_type {
        0 => ggml_type::GGML_TYPE_F32,
        1 => ggml_type::GGML_TYPE_F16,
        2 => ggml_type::GGML_TYPE_Q4_0,
        3 => ggml_type::GGML_TYPE_Q4_1,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        6 => ggml_type::GGML_TYPE_Q5_0,
        7 => ggml_type::GGML_TYPE_Q5_1,
        8 => ggml_type::GGML_TYPE_Q8_0,
        9 => ggml_type::GGML_TYPE_Q8_1,
        10 => ggml_type::GGML_TYPE_Q2_K,
        11 => ggml_type::GGML_TYPE_Q3_K,
        12 => ggml_type::GGML_TYPE_Q4_K,
        13 => ggml_type::GGML_TYPE_Q5_K,
        14 => ggml_type::GGML_TYPE_Q6_K,
        15 => ggml_type::GGML_TYPE_Q8_K,
        16 => ggml_type::GGML_TYPE_IQ2_XXS,
        17 => ggml_type::GGML_TYPE_IQ2_XS,
        18 => ggml_type::GGML_TYPE_IQ3_XXS,
        19 => ggml_type::GGML_TYPE_IQ1_S,
        20 => ggml_type::GGML_TYPE_IQ4_NL,
        21 => ggml_type::GGML_TYPE_IQ3_S,
        22 => ggml_type::GGML_TYPE_IQ2_S,
        23 => ggml_type::GGML_TYPE_IQ4_XS,
        24 => ggml_type::GGML_TYPE_I8,
        25 => ggml_type::GGML_TYPE_I16,
        26 => ggml_type::GGML_TYPE_I32,
        27 => ggml_type::GGML_TYPE_I64,
        28 => ggml_type::GGML_TYPE_F64,
        29 => ggml_type::GGML_TYPE_IQ1_M,
        30 => ggml_type::GGML_TYPE_BF16,
        _ => panic!("invalid ggml type {:?}", ggml_type),
    }
}

pub fn get_gguf_kv_type(kv_type: u32) -> gguf_type {
    match kv_type {
        0 => gguf_type::GGUF_TYPE_UINT8,
        1 => gguf_type::GGUF_TYPE_INT8,
        2 => gguf_type::GGUF_TYPE_UINT16,
        3 => gguf_type::GGUF_TYPE_INT16,
        4 => gguf_type::GGUF_TYPE_UINT32,
        5 => gguf_type::GGUF_TYPE_INT32,
        6 => gguf_type::GGUF_TYPE_FLOAT32,
        7 => gguf_type::GGUF_TYPE_BOOL,
        8 => gguf_type::GGUF_TYPE_STRING,
        9 => gguf_type::GGUF_TYPE_ARRAY,
        10 => gguf_type::GGUF_TYPE_UINT64,
        11 => gguf_type::GGUF_TYPE_INT64,
        12 => gguf_type::GGUF_TYPE_FLOAT64,
        _ => panic!("invalid kv type {:?}", kv_type),
    }
}

pub fn gguf_find_key(ctx: &gguf_context, key: &str) -> Option<usize> {
    let n_kv = ctx.kv.len();
    for i in 0..n_kv {
        if ctx.kv[i].key == key {
            return Some(i);
        }
    }
    None
}

pub fn gguf_get_val_u32(ctx: &gguf_context, key: &str) -> u32 {
    let key_id_result = gguf_find_key(&ctx, key);
    if let Some(key_id) = key_id_result {
        let value = &ctx.kv[key_id].value;
        if let gguf_value::uint32(i) = value {
            return *i;
        } else {
            panic!("invalid value type {:?}", value)
        }
    } else {
        panic!("invalid key {:?}", key);
    }
}

pub fn gguf_get_val_str(v: &gguf_value) -> &String {
    if let gguf_value::string(str) = v {
        return str;
    } else {
        panic!("invalid value type {:?}", v)
    }
}

pub fn gguf_get_val_i32(v: &gguf_value) -> i32 {
    if let gguf_value::int32(v) = v {
        return *v;
    } else {
        panic!("invalid value type {:?}", v)
    }
}

pub fn gguf_get_val_array_len(ctx: &gguf_context, key: &str) -> usize {
    let key_id_result = gguf_find_key(&ctx, key);
    if let Some(key_id) = key_id_result {
        let value = &ctx.kv[key_id].value;
        if let gguf_value::array(arr) = value {
            return arr.len();
        } else {
            panic!("invalid value type {:?}", value)
        }
    } else {
        panic!("invalid key {:?}", key);
    }
}

pub fn gguf_read_u32(buf: &[u8], offset: &mut usize) -> u32 {
    let v = LittleEndian::read_u32(&buf[*offset..*offset + 4 as usize]);
    *offset = *offset + 4;
    return v;
}

pub fn gguf_read_i32(buf: &[u8], offset: &mut usize) -> i32 {
    let v = LittleEndian::read_i32(&buf[*offset..*offset + 4 as usize]);
    *offset = *offset + 4;
    return v;
}

pub fn gguf_read_u64(buf: &[u8], offset: &mut usize) -> u64 {
    let v = LittleEndian::read_u64(&buf[*offset..*offset + 8 as usize]);
    *offset = *offset + 8;
    return v;
}

pub fn gguf_read_i64(buf: &[u8], offset: &mut usize) -> i64 {
    let v = LittleEndian::read_i64(&buf[*offset..*offset + 8 as usize]);
    *offset = *offset + 4;
    return v;
}

pub fn gguf_read_f32(buf: &[u8], offset: &mut usize) -> f32 {
    let v = LittleEndian::read_f32(&buf[*offset..*offset + 4 as usize]);
    *offset = *offset + 4;
    return v;
}

pub fn gguf_read_str(buf: &[u8], offset: &mut usize) -> String {
    let n = LittleEndian::read_u64(&buf[*offset..*offset + 8 as usize]);
    *offset = *offset + 8;
    let data = std::str::from_utf8(&buf[*offset..*offset + n as usize])
        .unwrap()
        .to_string();
    *offset = *offset + n as usize;
    return data;
}
