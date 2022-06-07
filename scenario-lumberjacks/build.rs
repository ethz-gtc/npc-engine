/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::process::Command;
use std::str;

fn main() {
    let output = Command::new("git")
        .args(&["describe", "--always", "--dirty"])
        .output()
        .unwrap();

    let git_hash = str::from_utf8(&*output.stdout).unwrap();
    println!("cargo:rustc-env=GIT_HASH={}", git_hash);
}
