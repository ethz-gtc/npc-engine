/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use ggez::graphics;
use ggez::graphics::{Canvas, Image};
use ggez::Context;
use image::png::PngEncoder;
use image::{ColorType, ImageBuffer, Rgba};

use crate::{config, output_path, PreWorldHookArgs, PreWorldHookFn, WorldGlobalState};

pub fn screenshot(
    ctx: &mut Context,
    world: &WorldGlobalState,
    assets: &BTreeMap<String, Image>,
    path: &str,
) {
    let canvas = Canvas::with_window_size(ctx).unwrap();
    graphics::set_canvas(ctx, Some(&canvas));
    graphics::clear(
        ctx,
        graphics::Color::new(
            config().display.background.0,
            config().display.background.1,
            config().display.background.2,
            1.,
        ),
    );

    world.draw(ctx, assets);

    graphics::present(ctx).unwrap();
    let image = canvas.image();

    let width = image.width() as u32;
    let height = image.height() as u32;
    let image_data: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(width, height, image.to_rgba8(ctx).unwrap()).unwrap();

    let flipped_image_data = image::imageops::flip_vertical(&image_data);

    let dir = {
        let mut path = PathBuf::from(path);
        path.pop();
        path.to_str().unwrap().to_owned()
    };
    fs::create_dir_all(dir).unwrap();

    let file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .unwrap();

    PngEncoder::new(file)
        .encode(&flipped_image_data, width, height, ColorType::Rgba8)
        .unwrap();
}

pub fn screenshot_hook() -> PreWorldHookFn {
    Box::new(
        |PreWorldHookArgs {
             world,
             ctx,
             assets,
             run,
             turn,
             ..
         }| {
            if let Some(ctx) = ctx {
                screenshot(
                    ctx,
                    world,
                    assets,
                    &format!(
                        "{}/{}/screenshots/turn{:06}.png",
                        output_path(),
                        run.map(|n| n.to_string()).unwrap_or_default(),
                        turn,
                    ),
                );
            }
        },
    )
}
