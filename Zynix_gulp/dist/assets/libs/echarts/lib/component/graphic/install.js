
/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/


/**
 * AUTO-GENERATED FILE. DO NOT MODIFY.
 */

/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
import { isArray } from 'zrender/lib/core/util.js';
import { GraphicComponentModel } from './GraphicModel.js';
import { GraphicComponentView } from './GraphicView.js';
export function install(registers) {
  registers.registerComponentModel(GraphicComponentModel);
  registers.registerComponentView(GraphicComponentView);
  registers.registerPreprocessor(function (option) {
    var graphicOption = option.graphic;
    // Convert
    // {graphic: [{left: 10, type: 'circle'}, ...]}
    // or
    // {graphic: {left: 10, type: 'circle'}}
    // to
    // {graphic: [{elements: [{left: 10, type: 'circle'}, ...]}]}
    if (isArray(graphicOption)) {
      if (!graphicOption[0] || !graphicOption[0].elements) {
        option.graphic = [{
          elements: graphicOption
        }];
      } else {
        // Only one graphic instance can be instantiated. (We don't
        // want that too many views are created in echarts._viewMap.)
        option.graphic = [option.graphic[0]];
      }
    } else if (graphicOption && !graphicOption.elements) {
      option.graphic = [{
        elements: [graphicOption]
      }];
    }
  });
}