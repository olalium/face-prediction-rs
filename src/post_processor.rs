use ndarray::s;
use ort::{tensor::OrtOwnedTensor, OrtError, Value};

pub type Bbox = [f32; 4];
pub type UltraResult = Vec<(Bbox, f32)>;

/// Positive additive constant to avoid divide-by-zero.
const EPS: f32 = 1.0e-7;

pub struct UltraOutput {
    pub bbox_with_confidences: UltraResult,
}

impl UltraOutput {
    pub fn new(outputs: Vec<Value>) -> Result<UltraOutput, OrtError> {
        let output_0: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
        let confidences_view = output_0.view();
        let confidences = confidences_view.slice(s![0, .., 1]);

        let output_1: OrtOwnedTensor<f32, _> = outputs[1].try_extract()?;
        let bbox_view = output_1.view();
        let bbox_arr = bbox_view.to_slice().unwrap().to_vec();
        let bboxes: Vec<Bbox> = bbox_arr.chunks(4).map(|x| x.try_into().unwrap()).collect();

        let mut bboxes_with_confidences: Vec<_> = bboxes
            .iter()
            .zip(confidences.iter())
            .filter_map(|(bbox, confidence)| match confidence {
                x if *x > 0.5 => Some((bbox, confidence)),
                _ => None,
            })
            .collect();

        bboxes_with_confidences.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let selected_bboxes = non_maximum_suppression(bboxes_with_confidences, 0.5);
        let selected_bboxes_top = selected_bboxes.to_vec();

        return Ok(UltraOutput {
            bbox_with_confidences: selected_bboxes_top,
        });
    }
}

pub struct ArcFaceOutput {
    pub embedding: Vec<f32>,
}

impl ArcFaceOutput {
    pub fn new(outputs: Vec<Value>) -> Result<ArcFaceOutput, OrtError> {
        let output_1: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
        let embeddings_view = output_1.view();
        let embeddings_arr = embeddings_view.to_slice().unwrap().to_vec();
        Ok(ArcFaceOutput {
            embedding: embeddings_arr,
        })
    }
}

/// Run non-maximum-suppression on candidate bounding boxes.
///
/// The pairs of bounding boxes with confidences have to be sorted in **ascending** order of
/// confidence because we want to `pop()` the most confident elements from the back.
///
/// Start with the most confident bounding box and iterate over all other bounding boxes in the
/// order of decreasing confidence. Grow the vector of selected bounding boxes by adding only those
/// candidates which do not have a IoU scores above `max_iou` with already chosen bounding boxes.
/// This iterates over all bounding boxes in `sorted_bboxes_with_confidences`. Any candidates with
/// scores generally too low to be considered should be filtered out before.
fn non_maximum_suppression(
    mut sorted_bboxes_with_confidences: Vec<(&Bbox, &f32)>,
    max_iou: f32,
) -> Vec<(Bbox, f32)> {
    let mut selected = vec![];
    'candidates: loop {
        // Get next most confident bbox from the back of ascending-sorted vector.
        // All boxes fulfill the minimum confidence criterium.
        match sorted_bboxes_with_confidences.pop() {
            Some((bbox, confidence)) => {
                // Check for overlap with any of the selected bboxes
                for (selected_bbox, _) in selected.iter() {
                    match iou(bbox, selected_bbox) {
                        x if x > max_iou => continue 'candidates,
                        _ => (),
                    }
                }

                // bbox has no large overlap with any of the selected ones, add it
                selected.push((*bbox, *confidence))
            }
            None => break 'candidates,
        }
    }

    selected
}

/// Calculate the intersection-over-union metric for two bounding boxes.
fn iou(bbox_a: &Bbox, bbox_b: &Bbox) -> f32 {
    // Calculate corner points of overlap box
    // If the boxes do not overlap, the corner-points will be ill defined, i.e. the top left
    // corner point will be below and to the right of the bottom right corner point. In this case,
    // the area will be zero.
    let overlap_box: Bbox = [
        f32::max(bbox_a[0], bbox_b[0]),
        f32::max(bbox_a[1], bbox_b[1]),
        f32::min(bbox_a[2], bbox_b[2]),
        f32::min(bbox_a[3], bbox_b[3]),
    ];

    let overlap_area = bbox_area(&overlap_box);

    // Avoid division-by-zero with `EPS`
    overlap_area / (bbox_area(bbox_a) + bbox_area(bbox_b) - overlap_area + EPS)
}

/// Calculate the area enclosed by a bounding box.
///
/// The bounding box is passed as four-element array defining two points:
/// `[x_top_left, y_top_left, x_bottom_right, y_bottom_right]`
/// If the bounding box is ill-defined by having the bottom-right point above/to the left of the
/// top-left point, the area is zero.
fn bbox_area(bbox: &Bbox) -> f32 {
    let width = bbox[3] - bbox[1];
    let height = bbox[2] - bbox[0];
    if width < 0.0 || height < 0.0 {
        // bbox is empty/undefined since the bottom-right corner is above the top left corner
        return 0.0;
    }

    width * height
}
