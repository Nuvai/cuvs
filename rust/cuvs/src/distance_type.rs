/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

pub type DistanceType = ffi::cuvsDistanceType;

/// Bitwise Hamming distance for binary vectors.
pub const BITWISE_HAMMING: DistanceType = ffi::cuvsDistanceType_CUVS_DISTANCE_BITWISE_HAMMING;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwise_hamming_equals_ffi() {
        // Verify our friendly constant matches the auto-generated FFI binding.
        assert_eq!(
            BITWISE_HAMMING,
            ffi::cuvsDistanceType_CUVS_DISTANCE_BITWISE_HAMMING,
            "BITWISE_HAMMING must match FFI-generated constant"
        );
    }

    #[test]
    fn test_bitwise_hamming_value() {
        // Sanity check: the C enum value for BITWISE_HAMMING is 20.
        assert_eq!(BITWISE_HAMMING, 20);
    }
}
