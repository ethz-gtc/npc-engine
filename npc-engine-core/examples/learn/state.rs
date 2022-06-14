/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct State {
    pub map: [u8; 14],
    pub wood_count: u8,
    pub agent_pos: u8,
}
impl State {
    // The number of trees as seen by the agent:
    // [sum very left, just left, cur pos, just right, sum very right]
    pub fn local_view(&self) -> [f32; 5] {
        let pos = self.agent_pos as usize;
        let len = self.map.len();
        let left_left = if pos > 1 {
            let sum: u8 = self.map.iter().take(pos - 1).sum();
            sum as f32
        } else {
            0.
        };
        let left = if pos > 0 {
            self.map[pos - 1] as f32
        } else {
            0.
        };
        let mid = self.map[pos] as f32;
        let right = if pos < len - 1 {
            self.map[pos + 1] as f32
        } else {
            0.
        };
        let right_right = if pos < len - 2 {
            let sum: u8 = self.map.iter().skip(pos + 2).sum();
            sum as f32
        } else {
            0.
        };
        [left_left, left, mid, right, right_right]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_local_view() {
        let mut state = State {
            map: [1, 3, 2, 1, 3, 2, 1, 0, 1, 3, 2, 0, 1, 3],
            wood_count: 0,
            agent_pos: 0,
        };
        assert_eq!(state.local_view(), [0., 0., 1., 3., 19.]);
        state.agent_pos = 1;
        assert_eq!(state.local_view(), [0., 1., 3., 2., 17.]);
        state.agent_pos = 3;
        assert_eq!(state.local_view(), [4., 2., 1., 3., 13.]);
        state.agent_pos = 12;
        assert_eq!(state.local_view(), [19., 0., 1., 3., 0.]);
        state.agent_pos = 13;
        assert_eq!(state.local_view(), [19., 1., 3., 0., 0.]);
    }
}
