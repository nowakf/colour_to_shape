use lazy_static::lazy_static;

use opencv::{ photo, highgui, imgproc, prelude::*, videoio, core::{self, extract_channel, Point2i, Vec4d, Size, CV_8U, BORDER_CONSTANT, Vec4f, VecN, Vector}, Result, types::VectorOfScalar };
use opencv::video::BackgroundSubtractorMOG2Trait;

const FUDGE: u8 = 1;
const WINDOW: usize = 6;

//calibration routine finds hues in picture
//characterize hues in picture (run on setup?)
fn hues(frame: &Mat) -> Vec<u8> {
    //TODO: use k means clustering to extract colours
    {0..25}.map(|n| n * 10).collect()
}


fn debug_display(frame: &Mat) -> Result<()> {
    lazy_static! {
        static ref WINDOW_NAME : &'static str = {
            let name = "disp";
            highgui::named_window(name, 1).unwrap();
            name
        };
    }
    highgui::imshow(*WINDOW_NAME, frame)
}

fn main() -> Result<()> {
    let white = Vec4d::from([255.0, 255.0, 255.0, 255.0]);
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    let e_kernel = imgproc::get_structuring_element(imgproc::MORPH_ELLIPSE, Size::new(2, 2), Point2i::new(0, 0))?;

    let mut contours = opencv::types::VectorOfVectorOfPoint::new();
    let ( mut hsv, mut mask, mut tmp, mut frame) = ( Mat::default(), Mat::default(), Mat::default(), Mat::default());

    let hues = hues(&Mat::default());

    let mut frames : Vector<Mat> = Vector::new();

    let mut ptr = 0;

    while cam.read(&mut frame)? {
        if frame.size()?.width == 0 { continue }
        
        let mut out = Mat::zeros(frame.rows(), frame.cols(), CV_8U)?.to_mat()?;

        imgproc::cvt_color(&frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

        let (top, bottom) = (
            Vector::from_slice(&[hues[ptr % hues.len()].wrapping_add(FUDGE), 255, 255]),
            Vector::from_slice(&[hues[ptr % hues.len()].wrapping_sub(FUDGE), 0, 0]),
        );


        core::in_range(&hsv, &bottom, &top, &mut mask)?;

        frames.push(mask.clone());

        if frames.len() > WINDOW { frames.remove(0); } else if frames.len() < WINDOW { continue }
        
        //examine techniques for removing noise here
        photo::fast_nl_means_denoising_multi(&frames, &mut tmp, WINDOW as i32 / 2, WINDOW as i32 / 2, 300.0, 7, 7)?;
        imgproc::threshold(&tmp, &mut mask, 100.0, 255.0, imgproc::THRESH_OTSU)?;
        imgproc::find_contours(&mask, &mut contours, imgproc::RETR_LIST, imgproc::CHAIN_APPROX_SIMPLE, Point2i::new(0,0))?;

        for i in 0..contours.len() as i32 {
            imgproc::draw_contours(&mut out, &contours, i, white,
                                   1, imgproc::LINE_AA, &0.0, 0, Point2i::new(0,0))?;
        }

        if highgui::poll_key()? > 0 {
            ptr += 1;
        }

        debug_display(&out)?;

    }
    Ok(())
}
